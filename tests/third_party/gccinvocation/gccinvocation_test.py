# Adapted from David Malcolm's gcc invocation library.
#
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
from compiler_gym.third_party.gccinvocation.gccinvocation import (
    GccInvocation,
    cmdline_to_argv,
)
from tests.test_main import main


def test_cmdline_to_argv_simple():
    argstr = (
        "gcc -o scripts/genksyms/genksyms"
        " scripts/genksyms/genksyms.o"
        " scripts/genksyms/parse.tab.o"
        " scripts/genksyms/lex.lex.o"
    )
    assert cmdline_to_argv(argstr) == [
        "gcc",
        "-o",
        "scripts/genksyms/genksyms",
        "scripts/genksyms/genksyms.o",
        "scripts/genksyms/parse.tab.o",
        "scripts/genksyms/lex.lex.o",
    ]


def test_cmdline_to_argv_quoted():
    # (heavily edited from a kernel build)
    argstr = (
        "cc1 -quiet"
        " -DCONFIG_AS_CFI_SIGNAL_FRAME=1"
        # Here's the awkward argument:
        ' -DIPATH_IDSTR="QLogic kernel.org driver"'
        " -DIPATH_KERN_TYPE=0 -DKBUILD_STR(s)=#s"
        " -fprofile-arcs -"
    )
    assert cmdline_to_argv(argstr) == [
        "cc1",
        "-quiet",
        "-DCONFIG_AS_CFI_SIGNAL_FRAME=1",
        '-DIPATH_IDSTR="QLogic kernel.org driver"',
        "-DIPATH_KERN_TYPE=0",
        "-DKBUILD_STR(s)=#s",
        "-fprofile-arcs",
        "-",
    ]


def test_parse_compile():
    args = (
        "gcc -pthread -fno-strict-aliasing -O2 -g -pipe -Wall"
        " -Wp,-D_FORTIFY_SOURCE=2 -fexceptions -fstack-protector"
        " --param=ssp-buffer-size=4 -m64 -mtune=generic -D_GNU_SOURCE"
        " -fPIC -fwrapv -DNDEBUG -O2 -g -pipe -Wall -Wp,-D_FORTIFY_SOURCE=2"
        " -fexceptions -fstack-protector --param=ssp-buffer-size=4 -m64"
        ' -mtune=generic -D_GNU_SOURCE -fPIC -fwrapv -fPIC -DVERSION="0.7"'
        " -I/usr/include/python2.7 -c python-ethtool/ethtool.c"
        " -o build/temp.linux-x86_64-2.7/python-ethtool/ethtool.o"
    ).split()
    gccinv = GccInvocation(args)
    assert gccinv.argv == args
    assert gccinv.executable == "gcc"
    assert gccinv.is_driver
    assert gccinv.sources == ["python-ethtool/ethtool.c"]
    assert gccinv.defines == ["_GNU_SOURCE", "NDEBUG", "_GNU_SOURCE", 'VERSION="0.7"']
    assert gccinv.includepaths == ["/usr/include/python2.7"]
    assert gccinv.otherargs == [
        "-pthread",
        "-fno-strict-aliasing",
        "-O2",
        "-g",
        "-pipe",
        "-Wall",
        "-Wp,-D_FORTIFY_SOURCE=2",
        "-fexceptions",
        "-fstack-protector",
        "--param=ssp-buffer-size=4",
        "-m64",
        "-mtune=generic",
        "-fPIC",
        "-fwrapv",
        "-O2",
        "-g",
        "-pipe",
        "-Wall",
        "-Wp,-D_FORTIFY_SOURCE=2",
        "-fexceptions",
        "-fstack-protector",
        "--param=ssp-buffer-size=4",
        "-m64",
        "-mtune=generic",
        "-fPIC",
        "-fwrapv",
        "-fPIC",
        "-c",
    ]


def test_parse_link():
    args = (
        "gcc -pthread -shared -Wl,-z,relro"
        " build/temp.linux-x86_64-2.7/python-ethtool/ethtool.o"
        " build/temp.linux-x86_64-2.7/python-ethtool/etherinfo.o"
        " build/temp.linux-x86_64-2.7/python-ethtool/etherinfo_obj.o"
        " build/temp.linux-x86_64-2.7/python-ethtool/etherinfo_ipv6_obj.o"
        " -L/usr/lib64 -lnl -lpython2.7"
        " -o build/lib.linux-x86_64-2.7/ethtool.so"
    ).split()
    gccinv = GccInvocation(args)
    assert gccinv.argv == args
    assert gccinv.executable == "gcc"
    assert gccinv.sources == [
        "build/temp.linux-x86_64-2.7/python-ethtool/ethtool.o",
        "build/temp.linux-x86_64-2.7/python-ethtool/etherinfo.o",
        "build/temp.linux-x86_64-2.7/python-ethtool/etherinfo_obj.o",
        "build/temp.linux-x86_64-2.7/python-ethtool/etherinfo_ipv6_obj.o",
    ]
    assert gccinv.defines == []
    assert gccinv.includepaths == []


def test_parse_cplusplus():
    args = (
        "/usr/bin/c++   -DPYSIDE_EXPORTS -DQT_GUI_LIB -DQT_CORE_LIB"
        " -DQT_NO_DEBUG -O2 -g -pipe -Wall -Wp,-D_FORTIFY_SOURCE=2"
        " -fexceptions -fstack-protector --param=ssp-buffer-size=4"
        "  -m64 -mtune=generic  -Wall -fvisibility=hidden"
        " -Wno-strict-aliasing -O3 -DNDEBUG -fPIC"
        " -I/usr/include/QtGui -I/usr/include/QtCore"
        " -I/builddir/build/BUILD/pyside-qt4.7+1.1.0/libpyside"
        " -I/usr/include/shiboken -I/usr/include/python2.7"
        "    -o CMakeFiles/pyside.dir/dynamicqmetaobject.cpp.o"
        " -c /builddir/build/BUILD/pyside-qt4.7+1.1.0/libpyside/dynamicqmetaobject.cpp"
    )
    gccinv = GccInvocation(args.split())
    assert gccinv.executable == "/usr/bin/c++"
    assert gccinv.progname == "c++"
    assert gccinv.is_driver
    assert gccinv.sources == [
        "/builddir/build/BUILD/pyside-qt4.7+1.1.0/libpyside/dynamicqmetaobject.cpp"
    ]
    assert "PYSIDE_EXPORTS" in gccinv.defines
    assert "NDEBUG" in gccinv.defines
    assert "/builddir/build/BUILD/pyside-qt4.7+1.1.0/libpyside" in gccinv.includepaths
    assert "--param=ssp-buffer-size=4" in gccinv.otherargs


def test_complex_invocation():
    # A command line taken from libreoffice/3.5.0.3/5.fc17/x86_64/build.log was:
    #   R=/builddir/build/BUILD && S=$R/libreoffice-3.5.0.3 && O=$S/solver/unxlngx6.pro && W=$S/workdir/unxlngx6.pro &&  mkdir -p $W/CxxObject/xml2cmp/source/support/ $W/Dep/CxxObject/xml2cmp/source/support/ && g++ -DCPPU_ENV=gcc3 -DENABLE_GRAPHITE -DENABLE_GTK -DENABLE_KDE4 -DGCC -DGXX_INCLUDE_PATH=/usr/include/c++/4.7.2 -DHAVE_GCC_VISIBILITY_FEATURE -DHAVE_THREADSAFE_STATICS -DLINUX -DNDEBUG -DOPTIMIZE -DOSL_DEBUG_LEVEL=0 -DPRODUCT -DSOLAR_JAVA -DSUPD=350 -DUNIX -DUNX -DVCL -DX86_64 -D_PTHREADS -D_REENTRANT   -Wall -Wendif-labels -Wextra -fmessage-length=0 -fno-common -pipe  -fPIC -Wshadow -Wsign-promo -Woverloaded-virtual -Wno-non-virtual-dtor  -fvisibility=hidden  -fvisibility-inlines-hidden  -std=c++0x  -ggdb2  -Wp,-D_FORTIFY_SOURCE=2 -fstack-protector --param=ssp-buffer-size=4 -m64 -mtune=generic -DEXCEPTIONS_ON -fexceptions -fno-enforce-eh-specs   -Wp,-D_FORTIFY_SOURCE=2 -fstack-protector --param=ssp-buffer-size=4 -m64 -mtune=generic -c $S/xml2cmp/source/support/cmdline.cxx -o $W/CxxObject/xml2cmp/source/support/cmdline.o -MMD -MT $W/CxxObject/xml2cmp/source/support/cmdline.o -MP -MF $W/Dep/CxxObject/xml2cmp/source/support/cmdline.d -I$S/xml2cmp/source/support/ -I$O/inc/stl -I$O/inc/external -I$O/inc -I$S/solenv/inc/unxlngx6 -I$S/solenv/inc -I$S/res -I/usr/lib/jvm/java-1.7.0-openjdk.x86_64/include -I/usr/lib/jvm/java-1.7.0-openjdk.x86_64/include/linux -I/usr/lib/jvm/java-1.7.0-openjdk.x86_64/include/native_threads/include
    args = (
        "g++ -DCPPU_ENV=gcc3 -DENABLE_GRAPHITE -DENABLE_GTK"
        " -DENABLE_KDE4 -DGCC -DGXX_INCLUDE_PATH=/usr/include/c++/4.7.2"
        " -DHAVE_GCC_VISIBILITY_FEATURE -DHAVE_THREADSAFE_STATICS"
        " -DLINUX -DNDEBUG -DOPTIMIZE -DOSL_DEBUG_LEVEL=0 -DPRODUCT"
        " -DSOLAR_JAVA -DSUPD=350 -DUNIX -DUNX -DVCL -DX86_64"
        " -D_PTHREADS -D_REENTRANT   -Wall -Wendif-labels -Wextra"
        " -fmessage-length=0 -fno-common -pipe  -fPIC -Wshadow"
        " -Wsign-promo -Woverloaded-virtual -Wno-non-virtual-dtor"
        "  -fvisibility=hidden  -fvisibility-inlines-hidden"
        "  -std=c++0x  -ggdb2  -Wp,-D_FORTIFY_SOURCE=2"
        " -fstack-protector --param=ssp-buffer-size=4 -m64"
        " -mtune=generic -DEXCEPTIONS_ON -fexceptions"
        " -fno-enforce-eh-specs   -Wp,-D_FORTIFY_SOURCE=2"
        " -fstack-protector --param=ssp-buffer-size=4 -m64"
        " -mtune=generic -c $S/xml2cmp/source/support/cmdline.cxx"
        " -o $W/CxxObject/xml2cmp/source/support/cmdline.o -MMD"
        " -MT $W/CxxObject/xml2cmp/source/support/cmdline.o -MP"
        " -MF $W/Dep/CxxObject/xml2cmp/source/support/cmdline.d"
        " -I$S/xml2cmp/source/support/ -I$O/inc/stl"
        " -I$O/inc/external -I$O/inc -I$S/solenv/inc/unxlngx6"
        " -I$S/solenv/inc -I$S/res"
        " -I/usr/lib/jvm/java-1.7.0-openjdk.x86_64/include"
        " -I/usr/lib/jvm/java-1.7.0-openjdk.x86_64/include/linux"
        " -I/usr/lib/jvm/java-1.7.0-openjdk.x86_64/include/native_threads/include"
    )
    # Expand the shell vars in the arguments:
    args = args.replace("$W", "$S/workdir/unxlngx6.pro")
    args = args.replace("$O", "$S/solver/unxlngx6.pro")
    args = args.replace("$S", "$R/libreoffice-3.5.0.3")
    args = args.replace("$R", "/builddir/build/BUILD")
    assert "$" not in args

    gccinv = GccInvocation(args.split())
    assert gccinv.executable == "g++"
    assert gccinv.sources == [
        "/builddir/build/BUILD/libreoffice-3.5.0.3/xml2cmp/source/support/cmdline.cxx"
    ]
    assert "CPPU_ENV=gcc3" in gccinv.defines
    assert "EXCEPTIONS_ON" in gccinv.defines
    assert (
        "/builddir/build/BUILD/libreoffice-3.5.0.3/solver/unxlngx6.pro/inc/stl"
        in gccinv.includepaths
    )
    assert (
        "/usr/lib/jvm/java-1.7.0-openjdk.x86_64/include/native_threads/include"
        in gccinv.includepaths
    )
    assert "-Wall" in gccinv.otherargs


def test_restrict_to_one_source():
    args = (
        "gcc -fPIC -shared -flto -flto-partition=none"
        " -Isomepath -DFOO"
        " -o output.o input-f.c input-g.c input-h.c"
    )
    gccinv = GccInvocation(args.split())
    assert gccinv.sources == ["input-f.c", "input-g.c", "input-h.c"]

    gccinv2 = gccinv.restrict_to_one_source("input-g.c")
    assert gccinv2.sources == ["input-g.c"]
    assert gccinv2.argv == [
        "gcc",
        "-DFOO",
        "-Isomepath",
        "-fPIC",
        "-shared",
        "-flto",
        "-flto-partition=none",
        "input-g.c",
    ]


def test_kernel_build():
    argstr = (
        "gcc -Wp,-MD,drivers/media/pci/mantis/.mantis_uart.o.d"
        " -nostdinc -isystem /usr/lib/gcc/x86_64-redhat-linux/4.4.7/include"
        " -I/home/david/linux-3.9.1/arch/x86/include"
        " -Iarch/x86/include/generated -Iinclude"
        " -I/home/david/linux-3.9.1/arch/x86/include/uapi"
        " -Iarch/x86/include/generated/uapi"
        " -I/home/david/linux-3.9.1/include/uapi"
        " -Iinclude/generated/uapi"
        " -include /home/david/linux-3.9.1/include/linux/kconfig.h"
        " -D__KERNEL__ -Wall -Wundef -Wstrict-prototypes"
        " -Wno-trigraphs -fno-strict-aliasing -fno-common"
        " -Werror-implicit-function-declaration"
        " -Wno-format-security -fno-delete-null-pointer-checks"
        " -Os -m64 -mtune=generic -mno-red-zone -mcmodel=kernel"
        " -funit-at-a-time -maccumulate-outgoing-args"
        " -fstack-protector -DCONFIG_AS_CFI=1"
        " -DCONFIG_AS_CFI_SIGNAL_FRAME=1"
        " -DCONFIG_AS_CFI_SECTIONS=1 -DCONFIG_AS_FXSAVEQ=1"
        " -DCONFIG_AS_AVX=1 -pipe -Wno-sign-compare"
        " -fno-asynchronous-unwind-tables -mno-sse -mno-mmx"
        " -mno-sse2 -mno-3dnow -mno-avx -fno-reorder-blocks"
        " -fno-ipa-cp-clone -Wframe-larger-than=2048"
        " -Wno-unused-but-set-variable -fno-omit-frame-pointer"
        " -fno-optimize-sibling-calls -g"
        " -femit-struct-debug-baseonly -fno-var-tracking -pg"
        " -fno-inline-functions-called-once"
        " -Wdeclaration-after-statement -Wno-pointer-sign"
        " -fno-strict-overflow -fconserve-stack"
        " -DCC_HAVE_ASM_GOTO -Idrivers/media/dvb-core/"
        " -Idrivers/media/dvb-frontends/ -fprofile-arcs"
        " -ftest-coverage -DKBUILD_STR(s)=#s"
        " -DKBUILD_BASENAME=KBUILD_STR(mantis_uart)"
        " -DKBUILD_MODNAME=KBUILD_STR(mantis_core) -c"
        " -o drivers/media/pci/mantis/.tmp_mantis_uart.o"
        " drivers/media/pci/mantis/mantis_uart.c"
    )
    gccinv = GccInvocation(argstr.split())
    assert gccinv.executable == "gcc"
    assert gccinv.progname == "gcc"
    assert gccinv.sources == ["drivers/media/pci/mantis/mantis_uart.c"]
    assert "__KERNEL__" in gccinv.defines
    assert "KBUILD_STR(s)=#s" in gccinv.defines


def test_kernel_cc1():
    argstr = (
        "/usr/libexec/gcc/x86_64-redhat-linux/4.4.7/cc1 -quiet"
        " -nostdinc"
        " -I/home/david/linux-3.9.1/arch/x86/include"
        " -Iarch/x86/include/generated -Iinclude"
        " -I/home/david/linux-3.9.1/arch/x86/include/uapi"
        " -Iarch/x86/include/generated/uapi"
        " -I/home/david/linux-3.9.1/include/uapi"
        " -Iinclude/generated/uapi -Idrivers/media/dvb-core/"
        " -Idrivers/media/dvb-frontends/ -D__KERNEL__"
        " -DCONFIG_AS_CFI=1 -DCONFIG_AS_CFI_SIGNAL_FRAME=1"
        " -DCONFIG_AS_CFI_SECTIONS=1 -DCONFIG_AS_FXSAVEQ=1"
        " -DCONFIG_AS_AVX=1 -DCC_HAVE_ASM_GOTO -DKBUILD_STR(s)=#s"
        " -DKBUILD_BASENAME=KBUILD_STR(mantis_uart)"
        " -DKBUILD_MODNAME=KBUILD_STR(mantis_core)"
        " -isystem /usr/lib/gcc/x86_64-redhat-linux/4.4.7/include"
        " -include /home/david/linux-3.9.1/include/linux/kconfig.h"
        " -MD drivers/media/pci/mantis/.mantis_uart.o.d"
        " drivers/media/pci/mantis/mantis_uart.c -quiet"
        " -dumpbase mantis_uart.c -m64 -mtune=generic"
        " -mno-red-zone -mcmodel=kernel -maccumulate-outgoing-args"
        " -mno-sse -mno-mmx -mno-sse2 -mno-3dnow -mno-avx"
        " -auxbase-strip drivers/media/pci/mantis/.tmp_mantis_uart.o"
        " -g -Os -Wall -Wundef -Wstrict-prototypes -Wno-trigraphs"
        " -Werror-implicit-function-declaration -Wno-format-security"
        " -Wno-sign-compare -Wframe-larger-than=2048"
        " -Wno-unused-but-set-variable -Wdeclaration-after-statement"
        " -Wno-pointer-sign -p -fno-strict-aliasing -fno-common"
        " -fno-delete-null-pointer-checks -funit-at-a-time"
        " -fstack-protector -fno-asynchronous-unwind-tables"
        " -fno-reorder-blocks -fno-ipa-cp-clone"
        " -fno-omit-frame-pointer -fno-optimize-sibling-calls"
        " -femit-struct-debug-baseonly -fno-var-tracking"
        " -fno-inline-functions-called-once -fno-strict-overflow"
        " -fconserve-stack -fprofile-arcs -ftest-coverage -o -"
    )
    gccinv = GccInvocation(argstr.split())
    assert gccinv.executable == "/usr/libexec/gcc/x86_64-redhat-linux/4.4.7/cc1"
    assert gccinv.progname == "cc1"
    assert not gccinv.is_driver
    assert gccinv.sources == ["drivers/media/pci/mantis/mantis_uart.c"]


def test_not_gcc():
    argstr = "objdump -h drivers/media/pci/mantis/.tmp_mantis_uart.o"
    gccinv = GccInvocation(argstr.split())
    assert gccinv.executable == "objdump"
    assert gccinv.progname == "objdump"
    assert not gccinv.is_driver


def test_dash_x():
    argstr = (
        "gcc -D__KERNEL__ -Wall -Wundef -Wstrict-prototypes"
        " -Wno-trigraphs -fno-strict-aliasing -fno-common"
        " -Werror-implicit-function-declaration"
        " -Wno-format-security -fno-delete-null-pointer-checks"
        " -Os -m64 -mno-sse -mpreferred-stack-boundary=3"
        " -c -x c /dev/null -o .20355.tmp"
    )
    gccinv = GccInvocation(argstr.split())
    assert gccinv.executable == "gcc"
    assert gccinv.sources == ["/dev/null"]


def test_pipes():
    argstr = (
        "gcc -D__KERNEL__ -S -x c -c -O0 -mcmodel=kernel" " -fstack-protector" " - -o -"
    )
    gccinv = GccInvocation(argstr.split())
    assert gccinv.sources == ["-"]


def test_print_file_name():
    argstr = "gcc -print-file-name=include"
    gccinv = GccInvocation(argstr.split())
    assert gccinv.sources == []
    assert "-print-file-name=include" in gccinv.otherargs


def test_collect2():
    # From a kernel build:
    argstr = (
        "/usr/libexec/gcc/x86_64-redhat-linux/4.4.7/collect2"
        " --eh-frame-hdr --build-id -m elf_x86_64"
        " --hash-style=gnu -dynamic-linker"
        " /lib64/ld-linux-x86-64.so.2 -o .20501.tmp"
        " -L/usr/lib/gcc/x86_64-redhat-linux/4.4.7"
        " -L/usr/lib/gcc/x86_64-redhat-linux/4.4.7"
        " -L/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../lib64"
        " -L/lib/../lib64 -L/usr/lib/../lib64"
        " -L/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../.."
        " --build-id /tmp/cckRREmI.o"
    )
    gccinv = GccInvocation(argstr.split())
    assert gccinv.progname == "collect2"
    assert not gccinv.is_driver
    assert gccinv.sources == []


def test_link():
    # From a kernel build:
    argstr = (
        "gcc -o scripts/genksyms/genksyms"
        " scripts/genksyms/genksyms.o"
        " scripts/genksyms/parse.tab.o"
        " scripts/genksyms/lex.lex.o"
    )
    gccinv = GccInvocation(argstr.split())
    assert gccinv.progname == "gcc"
    assert gccinv.sources == [
        "scripts/genksyms/genksyms.o",
        "scripts/genksyms/parse.tab.o",
        "scripts/genksyms/lex.lex.o",
    ]


def test_quoted_spaces():
    # Ensure we can handle spaces within a quoted argument
    argstr = (
        "/usr/libexec/gcc/x86_64-redhat-linux/4.4.7/cc1 -quiet"
        " -nostdinc"
        " -I/home/david/linux-3.9.1/arch/x86/include"
        " -Iarch/x86/include/generated -Iinclude"
        " -I/home/david/linux-3.9.1/arch/x86/include/uapi"
        " -Iarch/x86/include/generated/uapi"
        " -I/home/david/linux-3.9.1/include/uapi"
        " -Iinclude/generated/uapi -D__KERNEL__ -DCONFIG_AS_CFI=1"
        " -DCONFIG_AS_CFI_SIGNAL_FRAME=1"
        " -DCONFIG_AS_CFI_SECTIONS=1 -DCONFIG_AS_FXSAVEQ=1"
        " -DCONFIG_AS_AVX=1 -DCC_HAVE_ASM_GOTO"
        # Here's the awkward argument:
        ' -DIPATH_IDSTR="QLogic kernel.org driver"'
        " -DIPATH_KERN_TYPE=0 -DKBUILD_STR(s)=#s"
        " -DKBUILD_BASENAME=KBUILD_STR(ipath_cq)"
        " -DKBUILD_MODNAME=KBUILD_STR(ib_ipath)"
        " -isystem /usr/lib/gcc/x86_64-redhat-linux/4.4.7/include"
        " -include /home/david/linux-3.9.1/include/linux/kconfig.h"
        " -MD drivers/infiniband/hw/ipath/.ipath_cq.o.d"
        " drivers/infiniband/hw/ipath/ipath_cq.c"
        " -quiet -dumpbase ipath_cq.c -m64 -mtune=generic"
        " -mno-red-zone -mcmodel=kernel"
        " -maccumulate-outgoing-args -mno-sse -mno-mmx -mno-sse2"
        " -mno-3dnow -mno-avx -auxbase-strip"
        " drivers/infiniband/hw/ipath/.tmp_ipath_cq.o"
        " -g -Os -Wall -Wundef -Wstrict-prototypes"
        " -Wno-trigraphs -Werror-implicit-function-declaration"
        " -Wno-format-security -Wno-sign-compare"
        " -Wframe-larger-than=2048 -Wno-unused-but-set-variable"
        " -Wdeclaration-after-statement -Wno-pointer-sign -p"
        " -fno-strict-aliasing -fno-common"
        " -fno-delete-null-pointer-checks -funit-at-a-time"
        " -fstack-protector -fno-asynchronous-unwind-tables"
        " -fno-reorder-blocks -fno-ipa-cp-clone"
        " -fno-omit-frame-pointer -fno-optimize-sibling-calls"
        " -femit-struct-debug-baseonly -fno-var-tracking"
        " -fno-inline-functions-called-once"
        " -fno-strict-overflow -fconserve-stack"
        " -fprofile-arcs -ftest-coverage -o -"
    )
    gccinv = GccInvocation.from_cmdline(argstr)
    assert gccinv.sources == ["drivers/infiniband/hw/ipath/ipath_cq.c"]
    assert 'IPATH_IDSTR="QLogic kernel.org driver"' in gccinv.defines
    assert "KBUILD_STR(s)=#s" in gccinv.defines
    assert "KBUILD_BASENAME=KBUILD_STR(ipath_cq)" in gccinv.defines


def test_space_after_dash_D():
    # Note the space between the -D and its argument:
    argstr = (
        "gcc -c -x c -D __KERNEL__ -D SOME_OTHER_DEFINE /dev/null -o /tmp/ccqbm5As.s"
    )
    gccinv = GccInvocation.from_cmdline(argstr)
    assert gccinv.defines == ["__KERNEL__", "SOME_OTHER_DEFINE"]
    assert gccinv.sources == ["/dev/null"]


def test_space_after_dash_I():
    argstr = (
        "./install/libexec/gcc/x86_64-unknown-linux-gnu/4.9.0/cc1 -quiet"
        " -nostdinc"
        " -I somedir"
        " -I some/other/dir"
        " -D __KERNEL__"
        " -D CONFIG_AS_CFI=1"
        " -D CONFIG_AS_CFI_SIGNAL_FRAME=1"
        " -D KBUILD_STR(s)=#s"
        " -D KBUILD_BASENAME=KBUILD_STR(empty)"
        " -D KBUILD_MODNAME=KBUILD_STR(empty)"
        " scripts/mod/empty.c"
        " -o -"
    )
    gccinv = GccInvocation.from_cmdline(argstr)
    assert gccinv.defines == [
        "__KERNEL__",
        "CONFIG_AS_CFI=1",
        "CONFIG_AS_CFI_SIGNAL_FRAME=1",
        "KBUILD_STR(s)=#s",
        "KBUILD_BASENAME=KBUILD_STR(empty)",
        "KBUILD_MODNAME=KBUILD_STR(empty)",
    ]
    assert gccinv.sources == ["scripts/mod/empty.c"]


def test_space_after_dash_U():
    argstr = (
        "./install/libexec/gcc/x86_64-unknown-linux-gnu/4.9.0/cc1"
        " -E -lang-asm -quiet -nostdinc -C -C"
        "-P -P"
        " -U x86"
        " -isystem /some/dir"
        " -include /some/path/to/kconfig.h"
        " -MD arch/x86/vdso/.vdso.lds.d"
        " arch/x86/vdso/vdso.lds.S"
        " -o arch/x86/vdso/vdso.lds"
        " -mtune=generic -march=x86-64 -fno-directives-only"
    )
    gccinv = GccInvocation.from_cmdline(argstr)
    assert gccinv.sources == ["arch/x86/vdso/vdso.lds.S"]


def test_MD_without_arg():
    argstr = (
        "/usr/bin/gcc"
        " -Wp,-MD,arch/x86/purgatory/.purgatory.o.d"
        " -nostdinc"
        " -isystem"
        " /usr/lib/gcc/x86_64-redhat-linux/5.1.1/include"
        " -I./arch/x86/include"
        " -Iarch/x86/include/generated/uapi"
        " -Iarch/x86/include/generated"
        " -Iinclude"
        " -I./arch/x86/include/uapi"
        " -Iarch/x86/include/generated/uapi"
        " -I./include/uapi"
        " -Iinclude/generated/uapi"
        " -include"
        " ./include/linux/kconfig.h"
        " -D__KERNEL__"
        " -fno-strict-aliasing"
        " -Wall"
        " -Wstrict-prototypes"
        " -fno-zero-initialized-in-bss"
        " -fno-builtin"
        " -ffreestanding"
        " -c"
        " -MD"
        " -Os"
        " -mcmodel=large"
        " -m64"
        " -DKBUILD_STR(s)=#s"
        " -DKBUILD_BASENAME=KBUILD_STR(purgatory)"
        " -DKBUILD_MODNAME=KBUILD_STR(purgatory)"
        " -c"
        " -o"
        " arch/x86/purgatory/purgatory.o"
        " arch/x86/purgatory/purgatory.c"
    )
    gccinv = GccInvocation.from_cmdline(argstr)
    assert gccinv.sources == ["arch/x86/purgatory/purgatory.c"]


def test_openssl_invocation():
    argstr = (
        "/usr/bin/gcc"
        " -Werror"
        " -D"
        " OPENSSL_DOING_MAKEDEPEND"
        " -M"
        " -fPIC"
        " -DOPENSSL_PIC"
        " -DZLIB"
        " -DOPENSSL_THREADS"
        " -D_REENTRANT"
        " -DDSO_DLFCN"
        " -DHAVE_DLFCN_H"
        " -DKRB5_MIT"
        " -m64"
        " -DL_ENDIAN"
        " -DTERMIO"
        " -Wall"
        " -O2"
        " -g"
        " -pipe"
        " -Wall"
        " -Werror=format-security"
        " -Wp,-D_FORTIFY_SOURCE=2"
        " -fexceptions"
        " -fstack-protector-strong"
        " --param=ssp-buffer-size=4"
        " -grecord-gcc-switches"
        " -m64"
        " -mtune=generic"
        " -Wa,--noexecstack"
        " -DPURIFY"
        " -DOPENSSL_IA32_SSE2"
        " -DOPENSSL_BN_ASM_MONT"
        " -DOPENSSL_BN_ASM_MONT5"
        " -DOPENSSL_BN_ASM_GF2m"
        " -DSHA1_ASM"
        " -DSHA256_ASM"
        " -DSHA512_ASM"
        " -DMD5_ASM"
        " -DAES_ASM"
        " -DVPAES_ASM"
        " -DBSAES_ASM"
        " -DWHIRLPOOL_ASM"
        " -DGHASH_ASM"
        " -I."
        " -I.."
        " -I../include"
        " -DOPENSSL_NO_DEPRECATED"
        " -DOPENSSL_NO_EC2M"
        " -DOPENSSL_NO_EC_NISTP_64_GCC_128"
        " -DOPENSSL_NO_GMP"
        " -DOPENSSL_NO_GOST"
        " -DOPENSSL_NO_JPAKE"
        " -DOPENSSL_NO_MDC2"
        " -DOPENSSL_NO_RC5"
        " -DOPENSSL_NO_RSAX"
        " -DOPENSSL_NO_SCTP"
        " -DOPENSSL_NO_SRP"
        " -DOPENSSL_NO_STORE"
        " -DOPENSSL_NO_UNIT_TEST"
        " cryptlib.c"
        " mem.c"
        " mem_clr.c"
        " mem_dbg.c"
        " cversion.c"
        " ex_data.c"
        " cpt_err.c"
        " ebcdic.c"
        " uid.c"
        " o_time.c"
        " o_str.c"
        " o_dir.c"
        " o_fips.c"
        " o_init.c"
        " fips_ers.c"
    )
    gccinv = GccInvocation.from_cmdline(argstr)
    assert gccinv.sources == [
        "cryptlib.c",
        "mem.c",
        "mem_clr.c",
        "mem_dbg.c",
        "cversion.c",
        "ex_data.c",
        "cpt_err.c",
        "ebcdic.c",
        "uid.c",
        "o_time.c",
        "o_str.c",
        "o_dir.c",
        "o_fips.c",
        "o_init.c",
        "fips_ers.c",
    ]


if __name__ == "__main__":
    main()
