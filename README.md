# MATimers library

MATimers is a library of timers in hierarchical form. The goal is to capture the time spent in a scope/routine as many times as they are called.

## HOW TO USE IT

### Miniaml requirement 

Two instructions are required to use MATimers: 

```
MATimer::timers::initialize();

... code ...

MATimer::timers::finalize();
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
### Development state

| MATimers feature                 | State       |
|----------------------------------|-------------|
| Sequential                       | Done        |
| MPI                              | Done        |
| OpenMP                           | Done        |
| Hybrid                           | not planned |
| Unbalanced timers trees with MPI | Todo        |

## MATrace

MATimer provides other tools such as trace generation in paje format readable with VITE. You can access this feature with the namespace `MATimer::MATrace`.

### How to use it

MATrace `initialize` and `finalize` are respectively hidden in the MATimer `initialize` and `finalize`. MATrace feature furnishes two routines to capture a task: start and stop. The general way to use it is :

```
MATimer::MATrace::start()
do_something();
MATimer::MATrace::stop("kernel_name");
```

The `finalize` routine handles writing MATrace files. In an MPI context, all data are sent to the master process that writes the file.

### Development state

| MATrace feature  | State       |
|------------------|-------------|
| Sequential       | Done        |
| MPI              | Done        |
| OpenMP           | TODO        |
| Hybrid           | not planned |
| Color generation | TODO        |

