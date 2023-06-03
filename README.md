# MATools library

MATools is a library that offers various tools, including MATimers (timers in hierarchical form), MATrace (Trace generation for VITE), and MAMemory (memory footprint printing).

## MATimer

MATimers are designed to track the execution time of a scope/routine whenever they are called. They are organized in a tree structure. Additionally, MATimers are compatible with MPI. They provide information such as the minimum time, average time, maximum time, percentage of execution time, and imbalance for each scope/routine.

### Minimal requirement 

Two instructions are required to use MATimers: 

```
MATools::MATimer::initialize();

... code ...

MATools::MATimer::finalize();
```
By using these functions, the root MATimerNode is created, allowing you to capture the runtime of your application.

### MATimers API

At the beginning of your function/routine, you can use one of the following instructions to capture a section:

```
START_TIMER("section_name");
```

or

```
Catch_Time_Section("section_name");
```

For nested sections, you can use the macro START_TIMER or Catch_Time_Section with:

```
Catch_Nested_Time_Section("nested_section_name");
```

Please note the following limitations: only one instruction is allowed per scope, and these timers are not thread-safe.

Another approach to capturing a section is by using the chrono_section([&](args){...}) function, which returns the duration as a double. Additionally, you can add a chrono_section to the timers tree using the add_capture_chrono_section("name", [&](args){...}) function.

### MATimers Output

There are two options for outputs, a file or in a unix shell. By default, the timers are sorted based on their times within a given MATimerNode level.

#### File option

The MATimers write file routine generates a file named MATimers.number_of_threads.perf or MATimers.number_of_MPI.perf, which contains the recorded timers.

#### Shell option

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
#### MATimers Verbosity

The verbosity levels are defined during compilation. Here are the different levels and their descriptions:


| Level       | Description      |
|-------------|------------------|
| level1                       | TThis level displays the timer name based on the current MATimerNode level when the start_timer or catch_time_section is called.      |

Example output with verbosity level = 1 (Note: Only the master rank displays this information with MPI).

```
MATimers_LOG: MATimers initialization 
Verbosity_message:-- > func3
Verbosity_message:---- > func2
Verbosity_message:------ > func1
MATimers_LOG: MATimers finalization
```
### MATimers Options

In order to modify the behavior of MATimers, you can utilize the following options:
Disable printing the timetable

#### Disable printing the timetable

To prevent the timetable from being printed, this routine should be called within the MATools::Finalize() routine.

```
MATools::MATimer::Optional::disable_print_timetable();
```

#### Disable writing the timetable file

To prevent the timetable from being written to a file, this routine should be called within the `MATools::Finalize()` routine.

```
MATools::MATimer::Optional::disable_write_file();
```

#### Enable the full tree mode

This option should be activated when all MPI processes do not build the same timers tree, such as in a master/slave scheme. This routine should be called within the `MATools::Finalize()` routine.

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

MAMemory offers a flexible approach to incorporate memory checkpoints in order to track memory usage at various points in the code. This tool utilizes `rusage` for memory-related measurements. `rusage` is a structure in programming that provides information about resource usage by a process or thread.

### Usage

Unlike other tools, MAMemory does not require an initialize or finalize routine. To obtain the memory usage at a specific point, you can utilize the `MATools::MAMemory::print_memory_footprint` function. This function creates a temporary memory checkpoint and displays the total memory usage size.

### Status of developments 

| MAMemory feature                 | Status      |
|----------------------------------|-------------|
| Sequential                       | Done        |
| MPI                              | Done        |
| Collect checkpoints              | In progress |
| Add checkpoint names             | Todo        |


## MATrace

MATools provides additional tools, including trace generation in the paje format, which can be read with VITE. This feature can be accessed using the namespace `MATools::MATrace`. VITE is a visualization tool commonly used for analyzing and visualizing traces and performance data generated by parallel and distributed applications. It provides a graphical interface that allows users to explore and understand the behavior of their applications by visualizing the execution flow, communication patterns, and performance metrics.

### How to use it

The initialization and finalization routines for MATrace are hidden within the `MATools::initialize` and `MATools::finalize` functions, respectively. MATrace offers two routines, start and stop, to capture a task. The general approach for using MATrace is as follows:

```
MATools::MATrace::start()
do_something();
MATools::MATrace::stop("kernel_name");
```

The `finalize` routine is responsible for writing the MATrace files. In an MPI context, all the data is sent to the master process, which then writes the `MATrace.txt` text file.

### MATrace Options

#### Activate MATrace

The default mode of MATrace works in serial and MPI but this tool is disabled. MATrace can be activated with this routine:

```
MATools::MATrace::Optional::active_MATrace_mode();
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
