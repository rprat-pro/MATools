# MATools library

MATools is a library poviding tools such as MATimers (timers in hierarchical form) or MATrace (Trace generation for VITE). 

## MATimer

MATimer aimes to capture the time spent in a scope/routine as many times as they are called. Timer are represented with a tree. These Timers work with MPI too and min time, mean time, max time, % of runtime and imbalance are shown for each scope/routine.

### Minimal requirement 

Two instructions are required to use MATimers: 

```
MATools::MATimer::initialize();

... code ...

MATools::MATimer::finalize();
```

These functions create the root MATimerNode and capture your application runtime.

### Place your timers

At the beginning of your function/routine, put this instruction :

```
START_TIMER("section_name");
```

Limitation: only one instruction per scope.
Limitation: these timers are not thread-safe.

### Output

There are two possibilities for outputs, a file or in the terminal. By default, timers are sorted by their times for a given MATimerNode level.

#### File

The MATimers write file routine creates a file `MATimers.number_of_threads.perf` or `MATimers.number_of_MPI.perf` that contains your timers.

#### Shell

Example :

```
 |-- start timetable ---------------------------------------------------------------------|
 |    name                 |    number Of Calls |            max(s), |            part(%) |
 |----------------------------------------------------------------------------------------|
 | > root                  |                  1 |           0.000206 |         100.000000 |
 | |--> func5              |                  1 |           0.000160 |          77.587640 |
 |    |--> func4           |                  5 |           0.000153 |          74.576807 |
 |       |--> func3        |                 20 |           0.000139 |          67.707052 |
 |          |--> func2     |                 60 |           0.000100 |          48.768693 |
 |             |--> func1  |                120 |           0.000009 |           4.549541 |
 |-- end timetable -----------------------------------------------------------------------|
```

### MATimer Options

#### Do not print timetable

This routine has to be called by the MATools::Finalise() routine.

```
MATools::MATimer::disable_print_timetable();
```

#### Do not write timetable file

This routine has to be called by the MATools::Finalise() routine.

```
MATools::MATimer::disable_write_file();
```

#### Use the full tree mode

This option has to be activated if all mpi processes do not built the same timers tree. Example : Master/slave scheme. This routine has to be called by the MATools::Finalise() routine.

```
active_full_tree_mode();
```

### Status of developments 

| MATimers feature                 | Status      |
|----------------------------------|-------------|
| Sequential                       | Done        |
| MPI                              | Done        |
| OpenMP                           | TODO        |
| Hybrid                           | not planned |
| Unbalanced timers trees with MPI | Done        |

## MAMemory

MAMemory provides a flexible way to add memory checkpoints to print the memory usage at different points of the code. This tool is based on rusage.

### How to use it

This tool does not need an `initialize` or `finalize` routine. To get the memory usage at only one point, the `MATools::MAMemory::print_memory_footprint` creates a temporary memory checkpoint and prints the total memory usage size. 


### Status of developments 

| MAMemory feature                 | Status      |
|----------------------------------|-------------|
| Sequential                       | Done        |
| MPI                              | Done        |
| Collect checkpoints              | In progress |
| Add checkpoint names             | Todo        |


## MATrace

MATimer provides other tools such as trace generation in paje format readable with VITE. You can access this feature with the namespace `MATimer::MATrace`.

### How to use it

MATrace `initialize` and `finalize` are respectively hidden in the MATimer `initialize` and `finalize`. MATrace feature furnishes two routines to capture a task: start and stop. The general way to use it is :

```
MATools::MATrace::start()
do_something();
MATools::MATrace::stop("kernel_name");
```

The `finalize` routine handles writing MATrace files. In an MPI context, all data are sent to the master process that writes the MATrace.txt file.

### MATrace Options

#### Activate MATrace

The default mode of MATrace works in serial and MPI but this tool is disabled. MATrace can be activated with this routine:

```
MATools::MATrace::active_MATrace_mode();
```
This routine has to be called by the MATools::Finalise() routine.

#### Activate OpenMP mode

This is special mode of MATrace, tasks are labelled by a thread id instead of an mpi process id. We use a different way to capture chrono sections: 

```
#pragma omp parallel ...
{
 MATools::MATrace::omp_start()
 do_something();
 MATools::MATrace::omp_stop("kernel_name");
}
```
This mode can only be activated if the MATrace mode is used: 

```
MATools::MATrace::active_MATrace_mode();
MATools::MATrace::active_omp_mode();
```

These routines have to be called by the first omp_start() routine.

WARNING : The OpenMP mode does not work correctly with MPI. If you use MPI+OpenMP, the trace would be generated with all tasks and sent on the master node but, as tasks are labelled with threads id with this mode, tasks will be overlapped for a same thread id. 

REMARK : label could be : MPI_ID * NB_THREADS + THREAD_ID

### ### Status of developments 

| MATrace feature  | Status      |
|------------------|-------------|
| Sequential       | Done        |
| MPI              | Done        |
| OpenMP           | Done        |
| Hybrid           | Not planned |
| Default color    | Done        |

## Debugging tools

### write local timers tree for each MPI process

More details : test/test_mpi_debug_unbalanced_timers.cxx

```
MATools::MAOutputManager::write_debug_file();
```

Advice : if you are in a deadlock during the finalization function, you can disable printing and writting with the following instructions


```
MATools::MATimer::Optional::disable_print_timetable();
MATools::MATimer::Optional::disable_write_file();
```
