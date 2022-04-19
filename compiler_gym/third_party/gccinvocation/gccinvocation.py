#   Copyright 2013 David Malcolm <dmalcolm@redhat.com>
#   Copyright 2013 Red Hat, Inc.
#
#   This library is free software; you can redistribute it and/or
#   modify it under the terms of the GNU Lesser General Public
#   License as published by the Free Software Foundation; either
#   version 2.1 of the License, or (at your option) any later version.
#
#   This library is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#   Lesser General Public License for more details.
#
#   You should have received a copy of the GNU Lesser General Public
#   License along with this library; if not, write to the Free Software
#   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301
#   USA
import argparse
import os
from pathlib import Path


def cmdline_to_argv(cmdline):
    """
    Reconstruct an argv list from a cmdline string
    """
    # Convert the input str to a list of characters
    # and Quoted instances
    def iter_fragments():
        class Quoted:
            def __init__(self, quotechar, text):
                self.quotechar = quotechar
                self.text = text

            def __repr__(self):
                return "Quoted(%r, %r)" % (self.quotechar, self.text)

            def __str__(self):
                return "%s%s%s" % (self.quotechar, self.text, self.quotechar)

        for i, fragment in enumerate(cmdline.split('"')):
            if i % 2:
                # within a quoted section:
                yield Quoted('"', fragment)
            else:
                for ch in fragment:
                    yield ch

    # Now split these characters+Quoted by whitespace:
    result = []
    pending_arg = ""
    for fragment in iter_fragments():
        if fragment in (" ", "\t"):
            result.append(pending_arg)
            pending_arg = ""
        else:
            pending_arg += str(fragment)
    if pending_arg:
        result.append(pending_arg)
    return result


class GccInvocation:
    """
    Parse a command-line invocation of GCC and extract various options
    of interest
    """

    def __init__(self, argv):
        # Store the original argv for logging / debugging.
        self.original_argv = argv

        # Strip `-Xclang` arguments now because the hyphenated parameters
        # confuse argparse:
        sanitized_argv = argv.copy()
        for i in range(len(argv) - 2, -1, -1):
            if argv[i] == "-Xclang":
                del argv[i + 1]
                del argv[i]

        self.argv = argv

        self.executable = argv[0]
        self.progname = os.path.basename(self.executable)
        DRIVER_NAMES = (
            "c89",
            "c99",
            "cc",
            "gcc",
            "c++",
            "g++",
            "xgcc",
            "clang",
            "clang++",
        )
        self.is_driver = self.progname in DRIVER_NAMES
        self.sources = []
        self.defines = []
        self.includepaths = []
        self.otherargs = []

        if self.progname == "collect2":
            # collect2 appears to have a (mostly) different set of
            # arguments to the rest:
            return

        parser = argparse.ArgumentParser(add_help=False)

        def add_flag_opt(flag):
            parser.add_argument(flag, action="store_true")

        def add_opt_with_param(flag):
            parser.add_argument(flag, type=str)

        def add_opt_NoDriverArg(flag):
            if self.is_driver:
                add_flag_opt(flag)
            else:
                add_opt_with_param(flag)

        parser.add_argument("-o", type=str)

        parser.add_argument("-D", type=str, action="append", default=[])
        parser.add_argument("-U", type=str, action="append", default=[])
        parser.add_argument("-I", type=str, action="append", default=[])

        # Arguments that take a further param:
        parser.add_argument("-x", type=str)
        # (for now, drop them on the floor)

        # Arguments for dependency generation (in the order they appear
        # in gcc/c-family/c.opt)
        # (for now, drop them on the floor)
        add_flag_opt("-M")
        add_opt_NoDriverArg("-MD")
        add_opt_with_param("-MF")
        add_flag_opt("-MG")
        add_flag_opt("-MM")
        add_opt_NoDriverArg("-MMD")
        add_flag_opt("-MP")
        add_opt_with_param("-MQ")
        add_opt_with_param("-MT")

        # Additional arguments for clang:
        add_opt_with_param("-resource-dir")
        add_opt_with_param("-target")

        # Various other arguments that take a 2nd argument:
        for arg in [
            "-include",
            "-imacros",
            "-idirafter",
            "-iprefix",
            "-iwithprefix",
            "-iwithprefixbefore",
            "-isysroot",
            "-imultilib",
            "-isystem",
            "-iquote",
        ]:
            parser.add_argument(arg, type=str)
        # (for now, drop them on the floor)

        # Various arguments to cc1 etc that take a 2nd argument:
        for arg in ["-dumpbase", "-auxbase-strip"]:
            parser.add_argument(arg, type=str)
        # (for now, drop them on the floor)

        args, remainder = parser.parse_known_args(sanitized_argv[1:])

        self.parsed_args = args
        self.defines = args.D
        self.includepaths = args.I

        for arg in remainder:
            if arg.startswith("-") and arg != "-":
                self.otherargs.append(arg)
            else:
                self.sources.append(arg)

        # Determine the absolute path of the generated output.
        output = self.parsed_args.o or "a.out"
        self.output_path = Path(output).absolute()

    @classmethod
    def from_cmdline(cls, cmdline):
        return cls(cmdline_to_argv(cmdline))

    def __repr__(self):
        return (
            "GccInvocation(executable=%r, sources=%r,"
            " defines=%r, includepaths=%r, otherargs=%r)"
            % (
                self.executable,
                self.sources,
                self.defines,
                self.includepaths,
                self.otherargs,
            )
        )

    def restrict_to_one_source(self, source):
        """
        Make a new GccInvocation, preserving most arguments, but
        restricting the compilation to just the given source file
        """
        newargv = [self.executable]
        newargv += ["-D%s" % define for define in self.defines]
        newargv += ["-I%s" % include for include in self.includepaths]
        newargv += self.otherargs
        newargv += [source]
        return GccInvocation(newargv)
