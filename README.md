# MATimers

MATimers is a library of timers in hierarchical form. The goal is to capture the time spent by the same scope as many times as it is called.

## HOW TO USE IT

### General

Two instructions to use MATimers: 

```
MATimer::timers::init_timers();

... code ...

MATimer::timers::print_and_write_timers();
```

### Place your timers

At the begining of your function, put this instruction :

```
START_TIMER("name");
```

Limitation : only one instruction per scope.
Limitation : these timers are not thread-safe.

## Output

Two outputs :

### File

The MATimers write file routine creates a file MATimers.number_of_threads.perf that contains your timers.

### Shell

Example :

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

