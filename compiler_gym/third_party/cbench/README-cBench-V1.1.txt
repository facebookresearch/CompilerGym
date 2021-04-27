****************************************************************
Collective Benchmark (cBench)

cBench is a collection of open-source programs with multiple datasets
assembled by the community. The source code of individual programs is
simplified to ease portability. The behavior of individual programs
and datasets is analyzed and recorded in the Collective Optimization
Database (http://cTuning.org/cdatabase) to enable realistic benchmarking
and research on program and architecture optimization. cBench supports
multiple compilers (GCC, LLVM, Intel, GCC4CLI, Open64, PathScale) and
simulators.

cBench/cDatasets website (downloads and documentation):
 http://cTuning.org/cbench

cTuning/cBench/cDatasets mailing lists (feedback, comments and bug reports):
 http://cTuning.org/community

cBench is originall based on MiBench bechmark:
 http://www.eecs.umich.edu/mibench/index.html

cBench/cDatasets are used for training of MILEPOST GCC to learn good optimizations
across multiple programs, datasets, compilers and architectures, and correlate
them with program features and run-time behavior:

http://cTuning.org/milepost-gcc

****************************************************************
Author:

 cBench/cDatasets initiative has been started by Grigori Fursin
 in 2008/2009 as an extension to MiBench/MiDataSets.

 http://fursin.net/research

****************************************************************
License:
 This benchmark is free software; you can redistribute it and/or modify it
 under the terms of the GNU General Public License version 2 as published
 by the Free Software Foundation.

 This benchmark is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty
 of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 See the GNU General Public License for more details:
 http://www.gnu.org/copyleft/gpl.html.

 CBench is partially based on modified MiBench benchmark.

 Though we made an effort to include only copyright free benchmarks and datasets,
 mistakes are possible. In such case, please report this problem immediately at:
 http://cTuning.org/cbench

****************************************************************
Release history:

"CBench" release history:

 V1.1 March 15, 2010

      Grigori Fursin fixed problems with gsm and lame, and added
                     additional environment variable CCC_OPTS_ADD
                     that is used during both compilation and linking
                     to pass such parameters as -pg for profiling, for example.

 V1.0 (MiBench for MiDataSets V1.4 - this branch is now finished)

      Erven Rohou (IRISA) updated most of the benchmarks to remove unused headers
      and make them more compliant with standard C. The reason is to improve benchmarks
      portability and make them compile on different embedded architectures or
      with GCC4CIL.

      Grigori Fursin updated the following:
      * Added __compile script for each benchmark to select which Makefile.* to use.
      * Wrapper files fgg-wrap.c have been changed to loop-wrap.c.
      * Merged all files from _run directory into one file _ccc_info_datasets
      * Removed _run directory
      * Added scripts _ccc* to automate iterative compilation using Continuous Collective Compilation Framework
        (including checking the output for correctness)

      All further developments are moved to the Collective Benchmark
      (since we plan to add more open-source programs):
      http://cTuning.org/cbench

"MiBench for MiDataSets" release history:

 V1.3 January 25, 2008

      Kenneth Hoste (Ghent University) evaluated MiDataSets and
      provided a valuable feedback. Several programs have been slightly modified
      (source code or loop wrapper) either to remove small bugs or to keep benchmark
      execution time approximately 10 seconds on AMD Athlon 64 3700+ processors:

      * consumer_lame
      * consumer_tiff2bw
      * consumer_tiffdither
      * consumer_tiffmedian
      * office_ghostscript
      * office_rsynth

      Several datasets have also been modified/changed to work properly
      with the updated programs:

      * consumer_tiff_data (all ??.bw.tif have been converted to 8-bit grayscale
                           instead of 1-bit B&W to work properly with consumer_tiffdither)
      * office_data

 V1.2 November 19, 2007

      Qsort and stringsearch benchmarks are updated since they used
      standard library calls and were not useful for iterative optimizations -
      I added bodies of the qsort and strsearch functions to these benchmarks
      to have more room for optimizations.

      Note, that these benchmarks are now in the new directories:

      * automotive_qsort1
      * office_stringsearch1

      A few stupid mistakes are fixed in several benchmarks (security_blowfish_d,
      security_blowfish_e, security_pgp_d, security_pgp_e, dijkstra, patricia)
      where I used a file tmp-ccc-run-midataset instead of _finfo_dataset
      for the loop wrapper [thanks to Veerle Desmet from Ghent University].
      I am working on the Collective Compilation Framework and these files
      have been taken from that project by accident ...

      Some numbers for loop wrappers for dijkstra and patricia have been updated.

      A few tmp files have been removed (*.a, *.wav) - I forgot to remove
      them from the sources directories :( - it reduced the size of the tar ball by 40% !..

      Finally, Makefiles for Open64 compiler have been added.

 V1.1 September 05, 2007

      We would like to thank you all for your interest
      and valuable feedback, and are pleased to announce
      a bug-fix release for modified MiBench benchmark sources
      for MiDataSets V1. Any new benchmarks and datasets'
      contributions are welcome!

      The following benchmarks have been fixed:

      * consumer_lame
      * office_ghostscript
      * office_ispell
      * office_stringsearch
      * security_blowfish_d
      * security_blowfish_e
      * security_pgp_d
      * security_pgp_e

 V1.0 March 17, 2007

      First official release.

 V0.1 February 01, 2006

      Preliminary set of several datasets is prepared
      and used internally at INRIA for research.

      Started by Grigori Fursin and Olivier Temam
      (INRIA France and UNIDAPT Group)

      http://fursin.net/research
      http://unidapt.org

****************************************************************
Notes:

Most of the source codes have been slightly modified by Grigori Fursin
to simplify and automate iterative optimizations. A loop wrapper has been
added around the main procedure to make some benchmarks run longer when
real execution time is used for measurements instead of a simulator
(we do not yet take into account cache effects - it's left for the future work).
Each benchmark with each datasets run approximately 10 seconds
on INRIA cluster with AMD Athlon 64 3700+ processors.

Each directory has 4 Makefiles for GCC, Intel compilers, Open64 and PathScale compilers.
* Use "__compile" batch script with compiler name as the first parameter to compile the benchmark
  with a specific compiler, i.e. gcc, open64, pathscale or intel. In the second parameter
  you can specify optimization flags.
* Use "__run" batch script to execute a benchmark. The first
  parameter is the dataset number and the second optional parameter is the
  upper bound of the loop wrapper around the main procedure.
  If second parameter is omitted, the loop wrapper upper bound
  is taken from the file _ccc_info_datasets (for a given dataset number)

Each directory also has several scripts to automate iterative compilation using
Continuous Collective Compilation Framework (http://cTuning.org/ccc):
* "_ccc_prep" batch script if you need to prepare directory before compiling/running
  benchmark, i.e. copy some large files once, prepare libraries, etc...
* "_ccc_check_output.clean" - removes all output files that will be generated by the benchmark
* "_ccc_check_output.copy"  - save original output (when you run program for the first time
                              with the reference compilation
* "_ccc_check_output.diff"  - compare all produced output files with the saved ones.
                              Return NULL if files are the same.
* "_ccc_info_datasets"      - contains info about datasets:
                              <first line> - Max number of available datasets
                              ===== - separator
                              <dataset number>
                              <command line>
                              <loop wrapper>
                              =====
                              ...

* "_ccc_program_id"         - contains unique ID to be able to share optimization results with the community
                              in the Collective Optimization Database:
                              http://cTuning.org/cdatabase

Several batch files in the root directory are included as examples to automate iterative optimizations:
 all__create_work_dirs - creates temporal work directories for each benchmark
 all__delete_work_dirs - delete all temporal work directories
 all_compile - compile all benchmarks in the temporal work directories
 all_run__1_dataset - run all benchmarks with 1 dataset in the temporal work directories
 all_run__20_datasets - run all benchmarks with all datasets in the temporal work directories

Note: All benchmarks here have only 1 dataset. You can download extra cDatasets here:
      from http://cTuning.org/cbench.

You can find more info about how to use these benchmarks/datasets
in your research in the following publications:

http://unidapt.org/index.php/Dissemination#YYLP2010
http://unidapt.org/index.php/Dissemination#FT2009
http://unidapt.org/index.php/Dissemination#Fur2009
http://unidapt.org/index.php/Dissemination#FMTP2008
http://unidapt.org/index.php/Dissemination#FCOP2007

****************************************************************
Acknowledgments (suggestions, evaluation, bug fixes, etc):

 Erven Rohou (IRISA, France)
 Abdul Wahid Memon (Paris South University and INRIA, France)
 Menjato Rakoto (Paris South University and INRIA, France)
 Yang Chen (ICT, China)
 Yuanjie Huang (ICT, China)
 Chengyong Wu (ICT, China)
 Kenneth Hoste (Ghent University, Belgium)
 Veerle Desmet (Ghent University, Belgium)
 John Cavazos (University of Delaware, USA)
 Michael O'Boyle (University of Edinburgh, UK)
 Olivier Temam (INRIA Saclay, France) (original concept)
 Grigori Fursin (INRIA Saclay, France) (original concept)
****************************************************************
