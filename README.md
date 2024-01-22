# Bug Report

## GPU-related Bugs Overview

### Unique bugs reports of three GPU DBMSs:

| *DBMS*       | *Detected* | *Confirmed* | *Fixed*     |
| :----------- | :--------- | :---------- | :---------- |
| *PG-Strom*   | 26         | 26          | 26          |
| *HeavyDB*    | 12         | 5           | 0           |
| *Dask-sql*   | 14         | 3           | 1           |
| *Total*      | 52         | 34          | 27          |
| *Proportion* | -          | 65% (34/52) | 52% (27/52) |

### The statistics of the three types of GPU-related bugs:

| *DBMS*     | *Crash Bug* | *GPU Error Bug* | *GPU Logic Bug* |
| :--------- | :---------- | :-------------- | :-------------- |
| *HeteroDB* | 2           | 17              | 7               |
| *HeavyDB*  | 5           | 4               | 3               |
| *Dask-sql* | 1           | 8               | 5               |
| *Total*    | 8           | 29              | 15              |

## GPU-related Bugs Details

### **1. PG-Strom**

#### &#x20;#1 Crash Bug

*

&#x9;&#x20;

#### #2 Error Bug

#### &#x20;#3 Logic Bug

### **2. HeavyDB**

#### #1 ***Crash Bug***

*   ***(1)***  ***Crash Bug #1***

    ***URL:***  <https://github.com/heavyai/heavydb/issues/794#issue-1875017288>

    ***Brief Description:*** Create table with ( special \<property> = value ) causes crash, after set EXECUTOR\_DEVICE='GPU'.

    ***Status:***  Confirmed

    ***Test Case:***&#x20;

    ```sql
    -- SQL causes Crash Bug: 
    ALTER SESSION SET EXECUTOR_DEVICE='GPU';
    CREATE TABLE t0(c0 INT) WITH (PARTITIONS=0);
    CREATE TABLE t0(c0 INT) WITH (SORT_COLUMN=1);
    CREATE TABLE t0(c0 INT) WITH (VACUUM=1);
    CREATE TABLE t0(c0 INT) WITH (STORAGE_TYPE=1);

    -- Result:
    2023-08-31T08:20:55.953030 F 1188 17 3 ParserNode.h:2252 Check failed: val
    Stack trace:
    Parser::CreateTableStmt::executeDryRun
    Parser::CreateTableStmt::execute
    DdlCommandExecutor::execute
    DBHandler::executeDdl
    DBHandler::sql_execute_impl
    DBHandler::sql_execute_local
    DBHandler::sql_execute
    HeavyProcessor::process_sql_execute
    HeavyProcessor::dispatchCall
    TrackingProcessor::process
    apache::thrift::server::TConnectedClient::run
    apache::thrift::server::TThreadedServer::TConnectedClientRunner::run
    apache::thrift::concurrency::Thread::threadMain
    void std::__invoke_impl
    std::__invoke_result
    void std::thread::_Invoker
    std::thread::_Invoker
    std::thread::_State_impl
    clone
    /opt/heavyai/scripts/innerstartheavy: line 180:  1188 Aborted                 (core dumped) ./bin/heavydb $MAPD_DATA $RO --port $MAPD_TCP_PORT --http-port $MAPD_HTTP_PORT --calcite-port $MAPD_CALCITE_PORT $CONFIG $VERBOSE $*
    Failed to write to log, write storage/log/heavy_web_server.84a687ddbe10.root.log.ALL.20230831-075704.1189: file already closed
    startheavy 1177 exited
    ```

<!---->

*   ***(2) Crash Bug #2***

    ***URL:*** <https://github.com/heavyai/heavydb/issues/795>

    ***Brief Description:*** Insert values after create table with ( page\_size = value (too small or large) ) causes crash, after set EXECUTOR\_DEVICE='GPU'.

    ***Status:*** Confirmed

    ***Test Case:***

    ```sql
    -- SQL causes Crash Bug(1): 
    ALTER SESSION SET EXECUTOR_DEVICE='GPU';
    CREATE TABLE t1(c0 int) WITH (page_size=20); 
    INSERT INTO t1(c0) VALUES(1);

    -- Result:
    2023-08-31T08:26:35.743983 F 1460 18 2 FileBuffer.cpp:50 Check failed: pageSize_ > reservedHeaderSize_ (20 > 32)
    Stack trace:
    std::__cxx11::basic_string
    File_Namespace::FileBuffer::FileBuffer
    File_Namespace::FileMgr::createBufferUnlocked
    File_Namespace::FileMgr::createBuffer
    Data_Namespace::DataMgr::createChunkBuffer
    Chunk_NS::Chunk::createChunkBuffer
    Fragmenter_Namespace::InsertOrderFragmenter::createNewFragment
    Fragmenter_Namespace::InsertOrderFragmenter::insertDataImpl
    Fragmenter_Namespace::InsertOrderFragmenter::insertDataNoCheckpoint
    Fragmenter_Namespace::InsertDataLoader::insertData
    RelAlgExecutor::executeSimpleInsert
    Parser::InsertValuesStmt::execute
    DBHandler::sql_execute_impl
    DBHandler::sql_execute_local
    DBHandler::sql_execute
    HeavyProcessor::process_sql_execute
    HeavyProcessor::dispatchCall
    TrackingProcessor::process
    apache::thrift::server::TConnectedClient::run
    apache::thrift::server::TThreadedServer::TConnectedClientRunner::run
    apache::thrift::concurrency::Thread::threadMain
    void std::__invoke_impl
    std::__invoke_result
    void std::thread::_Invoker
    std::thread::_Invoker
    std::thread::_State_impl
    clone
    /opt/heavyai/scripts/innerstartheavy: line 180:  1460 Aborted                 (core dumped) ./bin/heavydb $MAPD_DATA $RO --port $MAPD_TCP_PORT --http-port $MAPD_HTTP_PORT --calcite-port $MAPD_CALCITE_PORT $CONFIG $VERBOSE $*
    Failed to write to log, write storage/log/heavy_web_server.84a687ddbe10.root.log.ALL.20230831-082529.1461: file already closed
    startheavy 1449 exited


    -- SQL causes Crash Bug(2): 
    ALTER SESSION SET EXECUTOR_DEVICE='GPU';
    CREATE TABLE t1(c0 INT) WITH (page_size=5441381524733052600); 
    INSERT INTO t1(c0) VALUES(1);

    -- Result:
    2023-08-31T08:33:55.148154 F 2248 24 2 File.cpp:74 Error trying to create file 'storage/data/table_1_9/0.18446744072721155768.data', file size 1 does not equal pageSize * numPages 18446743820680214528
    [2023-08-31 08:33:56.349176] [0x00007fe3bcff9000] [info]    25 4 DBHandler.cpp:7986 stdlog get_license_claims 25 21     
    [2023-08-31 08:33:56.422569] [0x00007fe3a6ffd000] [info]    26 5 DBHandler.cpp:934 stdlog_begin connect 26 0     
    [2023-08-31 08:33:56.730517] [0x00007fe3a6ffd000] [info]    26 5 DBHandler.cpp:980 User admin connected to database heavyai
    [2023-08-31 08:33:56.730554] [0x00007fe3a6ffd000] [info]    26 5 DBHandler.cpp:934 stdlog connect 26 308 heavyai admin 356-PMdU {"client","roles"} {"http:localhost:41314","super"}
    [2023-08-31 08:33:56.779125] [0x00007fe3a6ffd000] [info]    27 5 DBHandler.cpp:1275 stdlog get_session_info 27 0 heavyai admin 356-PMdU {"client"} {"http:localhost:41314"}
    [2023-08-31 08:33:56.920217] [0x00007fe3a6ffd000] [info]    28 5 DBHandler.cpp:1275 stdlog get_session_info 28 0 heavyai admin 356-PMdU {"client"} {"http:localhost:41314"}
    [2023-08-31 08:33:56.984323] [0x00007fe3a6ffd000] [info]    29 5 DBHandler.cpp:1275 stdlog get_session_info 29 0 heavyai admin 356-PMdU {"client"} {"http:172.17.0.1"}
    [2023-08-31 08:33:57.013509] [0x00007fe3a6ffd000] [info]    30 5 DBHandler.cpp:1182 stdlog get_status 30 0 heavyai admin 356-PMdU {"client"} {"http:172.17.0.1"}
    [2023-08-31 08:33:57.098269] [0x00007fe3a6ffd000] [info]    31 5 DBHandler.cpp:1230 stdlog get_hardware_info 31 0 heavyai admin 356-PMdU {"client"} {"http:172.17.0.1"}
    /opt/heavyai/scripts/innerstartheavy: line 180:  2248 Aborted                 (core dumped) ./bin/heavydb $MAPD_DATA $RO --port $MAPD_TCP_PORT --http-port $MAPD_HTTP_PORT --calcite-port $MAPD_CALCITE_PORT $CONFIG $VERBOSE $*
    Failed to write to log, write storage/log/heavy_web_server.84a687ddbe10.root.log.ALL.20230831-083200.2249: file already closed
    startheavy 2237 exited
    ```



*   ***(3) Crash Bug #3***

    ***URL:*** <https://github.com/heavyai/heavydb/issues/810>

    ***Brief Description:*** *SELECT \<column> FROM \<table> JOIN \<table> ON FALSE*  brings crash, after SET EXECUTOR\_DEVICE='GPU'.

    ***Status:*** Confirmed

    ***Test Case:***

    ```sql
    -- SQL causes Crash Bug: 
    ALTER SESSION SET EXECUTOR_DEVICE='GPU';
    CREATE TABLE t0(c0 INT);
    CREATE TABLE t1(c0 INT);
    SELECT t0.c0 FROM t0 JOIN t1 ON FALSE;

    -- Result:
    2023-10-02T11:17:11.259999 F 25 11 9 RelAlgOptimizer.cpp:1649 Check failed: literal->getType() == kBOOLEAN && literal->getVal<bool>()
    Stack trace:
    hoist_filter_cond_to_cross_join
    RelAlgDagBuilder::optimizeDag
    RelAlgDagBuilder::build
    RelAlgDagBuilder::buildDag
    DBHandler::execute_rel_alg
    QueryDispatchQueue::worker
    clone
    /opt/heavyai/scripts/innerstartheavy: line 180:    25 Aborted                 (core dumped) ./bin/heavydb $MAPD_DATA $RO --port $MAPD_TCP_PORT --http-port $MAPD_HTTP_PORT --calcite-port $MAPD_CALCITE_PORT $CONFIG $VERBOSE $*
    startheavy 9 exited
    ```



*   ***(4) Crash Bug #4***

    ***URL:*** <https://github.com/heavyai/heavydb/issues/811#issue-1948688857>

    ***Brief Description:*** *SELECT \* FROM \<table> JOIN \<table> ON CAST(\<number> AS BOOLEAN) WHERE FALSE*  brings crash, after SET EXECUTOR\_DEVICE='GPU'.

    ***Status:*** Confirmed

    ***Test Case:***

    ```sql
    -- SQL causes Crash Bug: 
    ALTER SESSION SET EXECUTOR_DEVICE='GPU';
    CREATE TABLE t0(c0 INT);
    CREATE TABLE t1(c0 INT);
    SELECT * FROM t1 JOIN t0 ON CAST(1 AS BOOLEAN) WHERE FALSE;

    -- Result:
    2023-10-18T02:58:46.458382 F 2867 11 9 RelLeftDeepInnerJoin.cpp:70 Check failed: dynamic_cast<const RexOperator*>(condition_.get())
    Stack trace:
    RelLeftDeepInnerJoin::RelLeftDeepInnerJoin
    create_left_deep_join
    RelAlgDagBuilder::optimizeDag
    RelAlgDagBuilder::build
    RelAlgDagBuilder::buildDag
    DBHandler::execute_rel_alg
    QueryDispatchQueue::worker
    clone
    /opt/heavyai/scripts/innerstartheavy: line 180:  2867 Aborted                 (core dumped) ./bin/heavydb $MAPD_DATA $RO --port $MAPD_TCP_PORT --http-port $MAPD_HTTP_PORT --calcite-port $MAPD_CALCITE_PORT $CONFIG $VERBOSE $*
    Failed to write to log, write storage/log/heavy_web_server.7ff8a2f8be28.root.log.ALL.20231018-025831.2868: file already closed
    startheavy 2856 exited
    ```



*   ***(5) Crash Bug #5***

    ***URL:*** <https://github.com/heavyai/heavydb/issues/812#issue-1948700610>

    ***Brief Description:*** *SELECT \* FROM \<table> JOIN \<table> ON NULL WHERE FALSE* brings crash, after SET EXECUTOR\_DEVICE='GPU'.

    ***Status:*** Confirmed

    ***Test Case:***

    ```sql
    -- SQL causes Crash Bug: 
    ALTER SESSION SET EXECUTOR_DEVICE='GPU';
    CREATE TABLE t0(c0 INT);
    CREATE TABLE t1(c0 INT);
    SELECT * FROM t1 JOIN t0 ON NULL WHERE FALSE;

    -- Result:
    2023-10-18T03:08:34.218366 F 3966 10 8 RelAlgDag.h:288 Check failed: ptr
    Stack trace:
    RelLeftDeepInnerJoin::RelLeftDeepInnerJoin
    create_left_deep_join
    RelAlgDagBuilder::optimizeDag
    RelAlgDagBuilder::build
    RelAlgDagBuilder::buildDag
    DBHandler::execute_rel_alg
    QueryDispatchQueue::worker
    clone
    /opt/heavyai/scripts/innerstartheavy: line 180:  3966 Aborted                 (core dumped) ./bin/heavydb $MAPD_DATA $RO --port $MAPD_TCP_PORT --http-port $MAPD_HTTP_PORT --calcite-port $MAPD_CALCITE_PORT $CONFIG $VERBOSE $*
    Failed to write to log, write storage/log/heavy_web_server.7ff8a2f8be28.root.log.ALL.20231018-030751.3967: file already closed
    startheavy 3955 exited
    ```



#### #2 Error Bug

*   ***(1)*** ***GPU Error Bug #1***

    ***URL:*** <https://github.com/heavyai/heavydb/issues/806#issue-1921478252>

    ***Brief Description:*** \<column> NOT IN \<column(overflow)> *in WHERE condition brings errors,  when set EXECUTOR\_DEVICE 'GPU'.

    ***Status:*** None

    ***Test Case:***

    ```sql
    CREATE TABLE IF NOT EXISTS t0(c0 INT, c1 INT);
    INSERT INTO t0(c1, c0) VALUES(-2016839045, 8138);

    -- SQL executes on the CPU: 
    ALTER SESSION SET EXECUTOR_DEVICE='CPU';
    SELECT t0.c0 FROM t0 WHERE (t0.c0) NOT IN (t0.c1+t0.c1, t0.c0);
    -- Result:
    No rows returned.

    -- SQL executes on the GPU: 
    ALTER SESSION SET EXECUTOR_DEVICE='GPU';
    SELECT t0.c0 FROM t0 WHERE (t0.c0) NOT IN (t0.c1+t0.c1, t0.c0);
    -- Result:
    ERR_OVERFLOW_OR_UNDERFLOW: Overflow or underflow.
    ```



*   ***(2)*** ***GPU Error Bug #2***

    ***URL:*** <https://github.com/heavyai/heavydb/issues/814#issue-1948719430>

    ***Brief Description:*** *SELECT \* FROM \<table> WHERE ((\<column> + \<column>) < \<column>) OR (\<column> = \<column>)* brings errors,  when set EXECUTOR\_DEVICE 'GPU'.

    ***Status:*** None

    ***Test Case:***

    ```sql
    CREATE TABLE t0(c0 INT);
    INSERT INTO t0(c0) VALUES(749466197), (-1013771122), (-1575001538);

    -- SQL executes on the CPU: 
    ALTER SESSION SET EXECUTOR_DEVICE='CPU';
    SELECT * FROM t0 WHERE ((t0.c0+t0.c0)<(t0.c0)) OR (t0.c0 = t0.c0);
    -- Result:
    c0
    749466197
    -1013771122

    -- SQL executes on the GPU: 
    ALTER SESSION SET EXECUTOR_DEVICE='GPU';
    SELECT * FROM t0 WHERE ((t0.c0+t0.c0)<(t0.c0)) OR (t0.c0 = t0.c0);
    -- Result:
    ERR_OVERFLOW_OR_UNDERFLOW: Overflow or underflow
    ```



<!---->

*   ***(3)*** ***GPU Error Bug #3***

    ***URL:*** <https://github.com/heavyai/heavydb/issues/816#issue-1978918534>

    ***Brief Description:*** *SELECT \* FROM \<table> JOIN ( SELECT ALL \<number> FROM \<table>) AS \<alias>*  brings errors,  when set EXECUTOR\_DEVICE 'GPU'.

    ***Status:*** None

    ***Test Case:***

    ```sql
    CREATE TABLE t1(c0 integer NOT NULL, c2 TEXT);

    -- SQL executes on the CPU: 
    ALTER SESSION SET EXECUTOR_DEVICE='CPU';
    SELECT * FROM t1 JOIN (SELECT ALL 0.42445757423087693 FROM t1) AS sub0 ON true WHERE NOT (t1.c2) NOT IN (t1.c2, t1.c2);
    -- Result:
    No rows returned.

    -- SQL executes on the GPU: 
    ALTER SESSION SET EXECUTOR_DEVICE='GPU';
    SELECT * FROM t1 JOIN (SELECT ALL 0.42445757423087693 FROM t1) AS sub0 ON true WHERE NOT (t1.c2) NOT IN (t1.c2, t1.c2);
    -- Result:
    Query execution failed because the query contains not supported self-join pattern. We suspect the query requires multiple left-deep join tree due to the join condition of the self-join and is not supported for now. Please consider chaning the table order in the FROM clause.
    ```



*   ***(4)*** ***GPU Error Bug #4***

    ***URL:*** <https://github.com/heavyai/heavydb/issues/817#issue-1978920821>

    ***Brief Description:*** *CAST(\<column>+\<column>(overflow) AS BOOLEAN)*  in WHERE condition brings errors,  when set EXECUTOR\_DEVICE 'GPU'.

    ***Status:*** None

    ***Test Case:***

    ```sql
    CREATE TABLE IF NOT EXISTS t0(c0 INTEGER NULL);
    INSERT INTO t0(c0) VALUES(1707341980);

    -- SQL executes on the CPU: 
    ALTER SESSION SET EXECUTOR_DEVICE='CPU';
    SELECT ALL t0.c0 FROM t0 WHERE CAST(t0.c0+t0.c0 AS BOOLEAN) IS FALSE IS NULL;
    -- Result:
    No rows returned.

    -- SQL executes on the GPU: 
    ALTER SESSION SET EXECUTOR_DEVICE='GPU';
    SELECT ALL t0.c0 FROM t0 WHERE CAST(t0.c0+t0.c0 AS BOOLEAN) IS FALSE IS NULL;
    -- Result:
    ERR_OVERFLOW_OR_UNDERFLOW: Overflow or underflow
    ```



#### #3 Logic Bug

*   ***(1)*** ***GPU Logic Bug #1***

    ***URL:*** <https://github.com/heavyai/heavydb/issues/804#issue-1921382141>

    ***Brief Description:***  *CAST('Infinity' AS INT)*  brings different results , when set EXECUTOR\_DEVICE 'CPU' and 'GPU'.

    ***Status:*** None

    ***Test Case:***

    ```sql
    CREATE TABLE IF NOT EXISTS t0(c0 DOUBLE  NULL);
    INSERT INTO t0(c0) VALUES(0), (1), ('Infinity');

    -- SQL executes on the CPU: 
    ALTER SESSION SET EXECUTOR_DEVICE='CPU';
    SELECT CAST(t0.c0 AS INT) FROM t0;
    -- Result:
    EXPR$0
    0
    1
    null

    -- SQL executes on the GPU: 
    ALTER SESSION SET EXECUTOR_DEVICE='GPU';
    SELECT CAST(t0.c0 AS INT) FROM t0;
    -- Result:
    EXPR$0
    0
    1
    2147483647
    ```



*   ***(2)*** ***GPU Logic Bug #2***

    ***URL:*** <https://github.com/heavyai/heavydb/issues/805#issue-1921443785>

    ***Brief Description:*** *SELECT DISTINCT \<column> FROM \<table> ORDER BY 1 DESC LIMIT 10*  brings different results , when set EXECUTOR\_DEVICE 'CPU' and 'GPU'.

    ***Status:*** None

    ***Test Case:***

    ```sql
    CREATE TABLE t0(c0 INT, c1 DOUBLE, SHARD KEY(c0)) WITH (shard_count=662);
    CREATE TABLE t1(c0 DECIMAL(2)  NULL, c1 FLOAT );
    INSERT INTO t0 VALUES(0,0),(-25151,null),(4115,-62),(4165,62),(-16256,null),(-115,-62),(11215,62);
    INSERT INTO t1 VALUES(1, 0.71818817),(0, -3.896708E7), (1, 0.9509316), (1, 8.043511E8),(1, 0.585598),(1,null),(null,9.1319142E8);

    -- SQL executes on the CPU: 
    ALTER SESSION SET EXECUTOR_DEVICE='CPU';
    SELECT DISTINCT t0.c0,t1.c1 FROM t0, t1  ORDER BY 1 desc limit 10;
    -- Result:
    c0|c1
    11215|9.131914e+08
    11215|0.585598
    11215|0.7181882
    11215|-3.896708e+07
    11215|8.043511e+08
    11215|NULL
    11215|0.9509316
    4165|9.131914e+08
    4165|0.7181882
    4165|NULL

    -- SQL executes on the GPU: 
    ALTER SESSION SET EXECUTOR_DEVICE='GPU'
    SELECT DISTINCT t0.c0,t1.c1 FROM t0, t1  ORDER BY 1 desc limit 10;
    -- Result:
    c0|c1
    4165|-3.896708e+07
    4165|0.7181882
    4165|0.9509316
    4165|8.043511e+08
    4115|0.9509316
    4115|8.043511e+08
    4115|NULL
    4115|0.7181882
    4115|0.585598
    4115|9.131914e+08
    ```



*   ***(3)*** ***GPU Logic Bug #3***

    ***URL:***&#x20;

    ***Brief Description:*** *SELECT DISTINCT \<column> FROM \<table> WHERE  CAST(\<column> AS INT) != 1*  brings different results , when set EXECUTOR\_DEVICE 'CPU' and 'GPU'.

    ***Status:*** None

    ***Test Case:***

    ```sql
    CREATE TABLE t0(c0 DECIMAL(2) NULL, c1 DOUBLE);
    INSERT INTO t0(c1) VALUES(0.20684618710775926),(1.7976931348623157E308);

    -- SQL executes on the CPU: 
    ALTER SESSION SET EXECUTOR_DEVICE='CPU';
    SELECT DISTINCT t0.c1 FROM t0 WHERE CAST(t0.c1 AS INT) != 1;
    -- Result:
    c1
    0.2068461871077593

    -- SQL executes on the GPU: 
    ALTER SESSION SET EXECUTOR_DEVICE='GPU';
    SELECT DISTINCT t0.c1 FROM t0 WHERE CAST(t0.c1 AS INT) != 1;
    -- Result:
    c1
    0.2068461871077593
    1.797693134862316e+308
    ```



### **3. Dask-sql**

#### &#x20;#1 Crash Bug

*   ***(1)*** ***Crash Bug #1***

    ***URL:*** <https://github.com/dask-contrib/dask-sql/issues/1227#issue-1903203722>

    ***Brief Description:***  *SELECT (((((NOT t1.c0))AND(('A' LIKE 'B' ESCAPE '/'))))=(t2.c0)) FROM t1, t2* causes crash.

    ***Status:***  None

    ***Test Case:***

    ```python
    import pandas as pd
    import dask.dataframe as dd
    from dask_sql import Context

    c = Context()

    df1 = pd.DataFrame({
        'c0': [5055.0],
        'c1': [False],
    })
    t1 = dd.from_pandas(df1, npartitions=1)
    c.create_table('t1', t1, gpu=False)
    c.create_table('t1_gpu', t1, gpu=True)

    df2 = pd.DataFrame({
        'c0': [True],
        'c1': ["'T'"],
    })
    t2 = dd.from_pandas(df2, npartitions=1)
    c.create_table('t2', t2, gpu=False)
    c.create_table('t2_gpu', t2, gpu=True)

    print('GPU Result:')
    result2= c.sql("SELECT (((((NOT t1_gpu.c0))AND(('A' LIKE 'B' ESCAPE '/'))))=(t2_gpu.c0)) FROM t1_gpu, t2_gpu").compute()
    print(result2)

    print('CPU Result:')
    result1= c.sql("SELECT (((((NOT t1.c0))AND(('A' LIKE 'B' ESCAPE '/'))))=(t2.c0)) FROM t1, t2").compute()
    print(result1)
    ```

    ```bash
    INFO:numba.cuda.cudadrv.driver:init
    GPU Result:
    WARNING:datafusion_optimizer.optimizer:Skipping optimizer rule 'simplify_expressions' due to unexpected error: Execution error: LIKE does not support escape_char
    WARNING:datafusion_optimizer.optimizer:Skipping optimizer rule 'simplify_expressions' due to unexpected error: Execution error: LIKE does not support escape_char
    WARNING:datafusion_optimizer.optimizer:Skipping optimizer rule 'simplify_expressions' due to unexpected error: Execution error: LIKE does not support escape_char
    WARNING:datafusion_optimizer.optimizer:Skipping optimizer rule 'simplify_expressions' due to unexpected error: Execution error: LIKE does not support escape_char
    WARNING:datafusion_optimizer.optimizer:Skipping optimizer rule 'simplify_expressions' due to unexpected error: Execution error: LIKE does not support escape_char
    WARNING:datafusion_optimizer.optimizer:Skipping optimizer rule 'simplify_expressions' due to unexpected error: Execution error: LIKE does not support escape_char
       NOT t1_gpu.c0 AND Utf8("A") LIKE Utf8("B") CHAR '/' = t2_gpu.c0
    0                                              False              
    CPU Result:
    WARNING:datafusion_optimizer.optimizer:Skipping optimizer rule 'simplify_expressions' due to unexpected error: Execution error: LIKE does not support escape_char
    WARNING:datafusion_optimizer.optimizer:Skipping optimizer rule 'simplify_expressions' due to unexpected error: Execution error: LIKE does not support escape_char
    WARNING:datafusion_optimizer.optimizer:Skipping optimizer rule 'simplify_expressions' due to unexpected error: Execution error: LIKE does not support escape_char
    WARNING:datafusion_optimizer.optimizer:Skipping optimizer rule 'simplify_expressions' due to unexpected error: Execution error: LIKE does not support escape_char
    WARNING:datafusion_optimizer.optimizer:Skipping optimizer rule 'simplify_expressions' due to unexpected error: Execution error: LIKE does not support escape_char
    WARNING:datafusion_optimizer.optimizer:Skipping optimizer rule 'simplify_expressions' due to unexpected error: Execution error: LIKE does not support escape_char
    Traceback (most recent call last):
      File "/tmp/bug.py", line 28, in <module>
        result1= c.sql("SELECT (((((NOT t1.c0))AND(('A' LIKE 'B' ESCAPE '/'))))=(t2.c0)) FROM t1, t2").compute()
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/base.py", line 314, in compute
        (result,) = compute(self, traverse=False, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/base.py", line 599, in compute
        results = schedule(dsk, keys, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/threaded.py", line 89, in get
        results = get_async(
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/local.py", line 511, in get_async
        raise_exception(exc, tb)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/local.py", line 319, in reraise
        raise exc
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/local.py", line 224, in execute_task
        result = _execute_task(task, data)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/core.py", line 119, in _execute_task
        return func(*(_execute_task(a, cache) for a in args))
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/optimization.py", line 990, in __call__
        return core.get(self.dsk, self.outkey, dict(zip(self.inkeys, args)))
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/core.py", line 149, in get
        result = _execute_task(task, cache)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/core.py", line 119, in _execute_task
        return func(*(_execute_task(a, cache) for a in args))
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/core.py", line 119, in <genexpr>
        return func(*(_execute_task(a, cache) for a in args))
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/core.py", line 119, in _execute_task
        return func(*(_execute_task(a, cache) for a in args))
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/core.py", line 119, in <genexpr>
        return func(*(_execute_task(a, cache) for a in args))
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/core.py", line 119, in _execute_task
        return func(*(_execute_task(a, cache) for a in args))
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/utils.py", line 73, in apply
        return func(*args, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/utils.py", line 1105, in __call__
        return getattr(__obj, self.method)(*args, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/pandas/core/generic.py", line 6240, in astype
        new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/pandas/core/internals/managers.py", line 448, in astype
        return self.apply("astype", dtype=dtype, copy=copy, errors=errors)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/pandas/core/internals/managers.py", line 352, in apply
        applied = getattr(b, f)(**kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/pandas/core/internals/blocks.py", line 526, in astype
        new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/pandas/core/dtypes/astype.py", line 299, in astype_array_safe
        new_values = astype_array(values, dtype, copy=copy)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/pandas/core/dtypes/astype.py", line 230, in astype_array
        values = astype_nansafe(values, dtype, copy=copy)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/pandas/core/dtypes/astype.py", line 95, in astype_nansafe
        return dtype.construct_array_type()._from_sequence(arr, dtype=dtype, copy=copy)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/pandas/core/arrays/masked.py", line 132, in _from_sequence
        values, mask = cls._coerce_to_array(scalars, dtype=dtype, copy=copy)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/pandas/core/arrays/boolean.py", line 344, in _coerce_to_array
        return coerce_to_array(value, copy=copy)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/pandas/core/arrays/boolean.py", line 185, in coerce_to_array
        raise TypeError("Need to pass bool-like values")
    TypeError: Need to pass bool-like values
    ```



#### #2 Error Bug

*   ***(1)*** ***GPU Error Bug #1***

    ***URL:*** <https://github.com/dask-contrib/dask-sql/issues/1231#issue-1904837968>

    ***Brief Description:*** *SELECT ((\<string>)!=(\<column>)) FROM \<tables>* brings errors,  when using 'GPU'.

    ***Status:*** Fixed

    ***Test Case:***

    ```python
    import pandas as pd
    import dask.dataframe as dd
    from dask_sql import Context

    c = Context()

    df0 = pd.DataFrame({
        'c0': [6464.0000]
    })
    t0 = dd.from_pandas(df0, npartitions=1)
    c.create_table('t0', t0, gpu=False)
    c.create_table('t0_gpu', t0, gpu=True)

    df1 = pd.DataFrame({
        'c0': [''],
    })
    t1 = dd.from_pandas(df1, npartitions=1)
    c.create_table('t1', t1, gpu=False)
    c.create_table('t1_gpu', t1, gpu=True)

    df3 = pd.DataFrame({
        'c0': [0.6918491708906861],
    })
    t3 = dd.from_pandas(df3, npartitions=1)
    c.create_table('t3', t3, gpu=False)
    c.create_table('t3_gpu', t3, gpu=True)

    print('CPU Result:')
    result1= c.sql("SELECT (('A')!=(t1.c0)) FROM t0, t1, t3").compute()
    print(result1)

    print('GPU Result:')
    result2= c.sql("SELECT (('A')!=(t1_gpu.c0)) FROM t0_gpu, t1_gpu, t3_gpu").compute()
    print(result2)
    ```

    ```bash
    INFO:numba.cuda.cudadrv.driver:init
    CPU Result:
       Utf8("A") != t1.c0
    0                True
    GPU Result:
    Traceback (most recent call last):
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/utils.py", line 193, in raise_on_meta_error
        yield
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/core.py", line 6470, in elemwise
        meta = partial_by_order(*parts, function=op, other=other)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/utils.py", line 1327, in partial_by_order
        return function(*args2, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/mixins/mixin_factory.py", line 11, in wrapper
        return method(self, *args1, *args2, **kwargs1, **kwargs2)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/indexed_frame.py", line 3350, in _binaryop
        ColumnAccessor(type(self)._colwise_binop(operands, op)),
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/nvtx/nvtx.py", line 101, in inner
        result = func(*args, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/frame.py", line 1750, in _colwise_binop
        else getattr(operator, fn)(left_column, right_column)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/mixins/mixin_factory.py", line 11, in wrapper
        return method(self, *args1, *args2, **kwargs1, **kwargs2)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/column/numerical.py", line 219, in _binaryop
        if (other := self._wrap_binop_normalization(other)) is NotImplemented:
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/column/column.py", line 607, in _wrap_binop_normalization
        return self.normalize_binop_value(other)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/column/numerical.py", line 312, in normalize_binop_value
        common_dtype = np.result_type(self.dtype, other)
      File "<__array_function__ internals>", line 200, in result_type
    TypeError: data type 'A' not understood

    The above exception was the direct cause of the following exception:

    Traceback (most recent call last):
      File "/tmp/bug.py", line 36, in <module>
        result2= c.sql("SELECT (('A')!=(t1_gpu.c0)) FROM t0_gpu, t1_gpu, t3_gpu").compute()
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/context.py", line 513, in sql
        return self._compute_table_from_rel(rel, return_futures)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/context.py", line 839, in _compute_table_from_rel
        dc = RelConverter.convert(rel, context=self)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/convert.py", line 61, in convert
        df = plugin_instance.convert(rel, context=context)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/logical/project.py", line 57, in convert
        new_columns[random_name] = RexConverter.convert(
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rex/convert.py", line 74, in convert
        df = plugin_instance.convert(rel, rex, dc, context=context)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rex/core/call.py", line 1129, in convert
        return operation(*operands, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rex/core/call.py", line 77, in __call__
        return self.f(*operands, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rex/core/call.py", line 140, in reduce
        return reduce(partial(self.operation, **kwargs), operands)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/core.py", line 1926, in <lambda>
        return lambda self, other: elemwise(op, self, other)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/core.py", line 6469, in elemwise
        with raise_on_meta_error(funcname(op)):
      File "/opt/conda/envs/rapids/lib/python3.10/contextlib.py", line 153, in __exit__
        self.gen.throw(typ, value, traceback)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/utils.py", line 214, in raise_on_meta_error
        raise ValueError(msg) from e
    ValueError: Metadata inference failed in `ne`.

    Original error is below:
    ------------------------
    TypeError("data type 'A' not understood")

    Traceback:
    ---------
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/utils.py", line 193, in raise_on_meta_error
        yield
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/core.py", line 6470, in elemwise
        meta = partial_by_order(*parts, function=op, other=other)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/utils.py", line 1327, in partial_by_order
        return function(*args2, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/mixins/mixin_factory.py", line 11, in wrapper
        return method(self, *args1, *args2, **kwargs1, **kwargs2)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/indexed_frame.py", line 3350, in _binaryop
        ColumnAccessor(type(self)._colwise_binop(operands, op)),
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/nvtx/nvtx.py", line 101, in inner
        result = func(*args, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/frame.py", line 1750, in _colwise_binop
        else getattr(operator, fn)(left_column, right_column)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/mixins/mixin_factory.py", line 11, in wrapper
        return method(self, *args1, *args2, **kwargs1, **kwargs2)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/column/numerical.py", line 219, in _binaryop
        if (other := self._wrap_binop_normalization(other)) is NotImplemented:
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/column/column.py", line 607, in _wrap_binop_normalization
        return self.normalize_binop_value(other)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/column/numerical.py", line 312, in normalize_binop_value
        common_dtype = np.result_type(self.dtype, other)
      File "<__array_function__ internals>", line 200, in result_type
    ```



*   ***(2)*** ***GPU Error Bug #2***

    ***URL:*** <https://github.com/dask-contrib/dask-sql/issues/1278#issue-2015427480>

    ***Brief Description:*** *SELECT \<number> FROM \<table> HAVING (\<TIMESTAMP> NOT BETWEEN \<TIMESTAMP> AND MAX(\<TIMESTAMP>))* brings errors,  when using 'GPU'.

    ***Status:*** Comfirmed

    ***Test Case:***

    ```python
    import pandas as pd
    import dask.dataframe as dd
    from dask_sql import Context

    c = Context()

    df0 = pd.DataFrame({
        'c0': ['CAST((12998) AS SMALLINT)'],
    })
    t0 = dd.from_pandas(df0, npartitions=1)
    c.create_table('t0', t0, gpu=False)
    c.create_table('t0_gpu', t0, gpu=True)

    print('Result1:')
    result1= c.sql("SELECT -2613 FROM t0 HAVING (TIMESTAMP '1991-02-28 13:42:12' NOT BETWEEN TIMESTAMP '1985-12-14 23:59:41' AND MAX(TIMESTAMP '2006-08-05 07:29:26'))").compute()
    print(result1)

    print('Result2:')
    result2= c.sql("SELECT -2613 FROM t0_gpu HAVING (TIMESTAMP '1991-02-28 13:42:12' NOT BETWEEN TIMESTAMP '1985-12-14 23:59:41' AND MAX(TIMESTAMP '2006-08-05 07:29:26'))").compute()
    print(result2)
    ```

    ```bash
    INFO:numba.cuda.cudadrv.driver:init
    Result1:
    Empty DataFrame
    Columns: [Int64(-2613)]
    Index: []
    Result2:
    Traceback (most recent call last):
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/utils.py", line 193, in raise_on_meta_error
        yield
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/core.py", line 6470, in elemwise
        meta = partial_by_order(*parts, function=op, other=other)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/utils.py", line 1327, in partial_by_order
        return function(*args2, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/indexed_frame.py", line 3375, in __array_ufunc__
        ret = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/frame.py", line 1761, in __array_ufunc__
        return _array_ufunc(self, ufunc, method, inputs, kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/utils/utils.py", line 93, in _array_ufunc
        return getattr(obj, op)(other)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/mixins/mixin_factory.py", line 11, in wrapper
        return method(self, *args1, *args2, **kwargs1, **kwargs2)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/indexed_frame.py", line 3350, in _binaryop
        ColumnAccessor(type(self)._colwise_binop(operands, op)),
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/nvtx/nvtx.py", line 101, in inner
        result = func(*args, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/frame.py", line 1750, in _colwise_binop
        else getattr(operator, fn)(left_column, right_column)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/mixins/mixin_factory.py", line 11, in wrapper
        return method(self, *args1, *args2, **kwargs1, **kwargs2)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/column/datetime.py", line 405, in _binaryop
        other = self._wrap_binop_normalization(other)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/column/column.py", line 606, in _wrap_binop_normalization
        other = other.dtype.type(other.item())
    ValueError: Converting an integer to a NumPy datetime requires a specified unit

    The above exception was the direct cause of the following exception:

    Traceback (most recent call last):
      File "/tmp/bug19/bug19.py", line 19, in <module>
        result2= c.sql("SELECT -2613 FROM t0_gpu HAVING (TIMESTAMP '1991-02-28 13:42:12' NOT BETWEEN TIMESTAMP '1985-12-14 23:59:41' AND MAX(TIMESTAMP '2006-08-05 07:29:26'))").compute()
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/context.py", line 513, in sql
        return self._compute_table_from_rel(rel, return_futures)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/context.py", line 839, in _compute_table_from_rel
        dc = RelConverter.convert(rel, context=self)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/convert.py", line 61, in convert
        df = plugin_instance.convert(rel, context=context)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/logical/project.py", line 28, in convert
        (dc,) = self.assert_inputs(rel, 1, context)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/base.py", line 84, in assert_inputs
        return [RelConverter.convert(input_rel, context) for input_rel in input_rels]
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/base.py", line 84, in <listcomp>
        return [RelConverter.convert(input_rel, context) for input_rel in input_rels]
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/convert.py", line 61, in convert
        df = plugin_instance.convert(rel, context=context)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/logical/filter.py", line 65, in convert
        df_condition = RexConverter.convert(rel, condition, dc, context=context)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rex/convert.py", line 74, in convert
        df = plugin_instance.convert(rel, rex, dc, context=context)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rex/core/call.py", line 1129, in convert
        return operation(*operands, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rex/core/call.py", line 77, in __call__
        return self.f(*operands, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rex/core/call.py", line 140, in reduce
        return reduce(partial(self.operation, **kwargs), operands)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/core.py", line 617, in __array_ufunc__
        return elemwise(numpy_ufunc, *inputs, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/core.py", line 6469, in elemwise
        with raise_on_meta_error(funcname(op)):
      File "/opt/conda/envs/rapids/lib/python3.10/contextlib.py", line 153, in __exit__
        self.gen.throw(typ, value, traceback)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/utils.py", line 214, in raise_on_meta_error
        raise ValueError(msg) from e
    ValueError: Metadata inference failed in `greater`.

    Original error is below:
    ------------------------
    ValueError('Converting an integer to a NumPy datetime requires a specified unit')

    Traceback:
    ---------
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/utils.py", line 193, in raise_on_meta_error
        yield
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/core.py", line 6470, in elemwise
        meta = partial_by_order(*parts, function=op, other=other)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/utils.py", line 1327, in partial_by_order
        return function(*args2, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/indexed_frame.py", line 3375, in __array_ufunc__
        ret = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/frame.py", line 1761, in __array_ufunc__
        return _array_ufunc(self, ufunc, method, inputs, kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/utils/utils.py", line 93, in _array_ufunc
        return getattr(obj, op)(other)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/mixins/mixin_factory.py", line 11, in wrapper
        return method(self, *args1, *args2, **kwargs1, **kwargs2)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/indexed_frame.py", line 3350, in _binaryop
        ColumnAccessor(type(self)._colwise_binop(operands, op)),
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/nvtx/nvtx.py", line 101, in inner
        result = func(*args, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/frame.py", line 1750, in _colwise_binop
        else getattr(operator, fn)(left_column, right_column)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/mixins/mixin_factory.py", line 11, in wrapper
        return method(self, *args1, *args2, **kwargs1, **kwargs2)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/column/datetime.py", line 405, in _binaryop
        other = self._wrap_binop_normalization(other)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/column/column.py", line 606, in _wrap_binop_normalization
        other = other.dtype.type(other.item())
    ```



*   ***(3)*** ***GPU Error Bug #3***

    ***URL:*** <https://github.com/dask-contrib/dask-sql/issues/1232#issue-1904872325>

    ***Brief Description:*** SELECT (CASE false WHEN (true) THEN (true) WHEN false THEN false END ), ('\<string>'), false FROM \<table>* *brings errors,  when using 'GPU'.

    ***Status:*** None

    ***Test Case:***

    ```python
    import pandas as pd
    import dask.dataframe as dd
    from dask_sql import Context

    c = Context()

    df0 = pd.DataFrame({
        'c0': [835.0000],
    })
    t0 = dd.from_pandas(df0, npartitions=1)

    c.create_table('t0', t0, gpu=False)
    c.create_table('t0_gpu', t0, gpu=True)

    print('CPU Result:')
    result1= c.sql("SELECT (CASE false WHEN (true) THEN (true) WHEN false THEN false END ), ('A'), false FROM t0").compute()
    print(result1)

    print('GPU Result:')
    result2= c.sql("SELECT (CASE false WHEN (true) THEN (true) WHEN false THEN false END ), ('A'), false FROM t0_gpu").compute()
    print(result2)
    ```

    ```bash
    INFO:numba.cuda.cudadrv.driver:init
    CPU Result:
       CASE Boolean(false) WHEN Boolean(true) THEN Boolean(true) WHEN Boolean(false) THEN Boolean(false) END Utf8("A")  Boolean(false)
    0                                              False                                                             A           False
    GPU Result:
    Traceback (most recent call last):
      File "/tmp/bug.py", line 20, in <module>
        result2= c.sql("SELECT (CASE false WHEN (true) THEN (true) WHEN false THEN false END ), ('A'), false FROM t0_gpu").compute()
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/context.py", line 513, in sql
        return self._compute_table_from_rel(rel, return_futures)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/context.py", line 869, in _compute_table_from_rel
        df = dc.assign()
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/datacontainer.py", line 229, in assign
        df.columns = self.column_container.columns
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/core.py", line 4887, in __setattr__
        object.__setattr__(self, key, value)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/core.py", line 4746, in columns
        renamed = _rename_dask(self, columns)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/core.py", line 7084, in _rename_dask
        metadata = _rename(names, df._meta)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/core.py", line 7055, in _rename
        df.columns = columns
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/dataframe.py", line 1094, in __setattr__
        super().__setattr__(key, col)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/nvtx/nvtx.py", line 101, in inner
        result = func(*args, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/dataframe.py", line 2473, in columns
        raise ValueError(
    ValueError: Length mismatch: expected 2 elements, got 3 elements 
    ```



*   ***(4)*** ***GPU Error Bug #4***

    ***URL:*** <https://github.com/dask-contrib/dask-sql/issues/1230>

    ***Brief Description:*** *SELECT ((\<string>)LIKE(\<column>)) FROM \<table>* brings errors,  when using 'GPU'.

    ***Status:*** None

    ***Test Case:***

    ```python
    import pandas as pd
    import dask.dataframe as dd
    from dask_sql import Context

    c = Context()

    df1 = pd.DataFrame({
        'c0': [0.6926717947094722],
        'c1': ['B'],
    })
    t1 = dd.from_pandas(df1, npartitions=1)

    c.create_table('t1', t1, gpu=False)
    c.create_table('t1_gpu', t1, gpu=True)

    print('CPU Result:')
    result1= c.sql("SELECT (('A')LIKE(t1.c1)) FROM t1").compute()
    print(result1)

    print('GPU Result:')
    result2= c.sql("SELECT (('A')LIKE(t1_gpu.c1)) FROM t1_gpu").compute()
    print(result2)
    ```

    ```bash
    INFO:numba.cuda.cudadrv.driver:init
    CPU Result:
       Utf8("A") LIKE t1.c1 
    0                  False
    GPU Result:
    Traceback (most recent call last):
      File "/tmp/bug.py", line 21, in <module>
        result2= c.sql("SELECT (('A')LIKE(t1_gpu.c1)) FROM t1_gpu").compute()
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/context.py", line 513, in sql
        return self._compute_table_from_rel(rel, return_futures)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/context.py", line 839, in _compute_table_from_rel
        dc = RelConverter.convert(rel, context=self)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/convert.py", line 61, in convert
        df = plugin_instance.convert(rel, context=context)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/logical/project.py", line 57, in convert
        new_columns[random_name] = RexConverter.convert(
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rex/convert.py", line 74, in convert
        df = plugin_instance.convert(rel, rex, dc, context=context)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rex/core/call.py", line 1129, in convert
        return operation(*operands, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rex/core/call.py", line 77, in __call__
        return self.f(*operands, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rex/core/call.py", line 402, in regex
        for char in regex:
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/core.py", line 3997, in __iter__
        yield from s
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/utils/utils.py", line 288, in __iter__
        raise TypeError(
    TypeError: Series object is not iterable. Consider using `.to_arrow()`, `.to_pandas()` or `.values_host` if you wish to iterate over the values.
    ```



*   ***(5)*** ***GPU Error Bug #5***

    ***URL:*** <https://github.com/dask-contrib/dask-sql/issues/1229#issue-1904417430>

    ***Brief Description:*** *SELECT \<number> FROM \<table> HAVING EVERY(\<boolean>)* brings errors,  when using 'GPU'.

    ***Status:*** None

    ***Test Case:***

    ```python
    import pandas as pd
    import dask.dataframe as dd
    from dask_sql import Context

    c = Context()

    df0 = pd.DataFrame({
        'c0': ['A'],
        'c1': ['B'],
    })
    t0 = dd.from_pandas(df0, npartitions=1)

    c.create_table('t0', t0, gpu=False)
    c.create_table('t0_gpu', t0, gpu=True)

    print('CPU Result:')
    result1= c.sql("SELECT 1 FROM t0 HAVING EVERY(true)").compute()
    print(result1)

    print('GPU Result:')
    result2= c.sql("SELECT 1 FROM t0_gpu HAVING EVERY(true)").compute()
    print(result2)
    ```

    ```bash
    INFO:numba.cuda.cudadrv.driver:init
    CPU Result:
       Int64(1)
    0         1
    GPU Result:
    Traceback (most recent call last):
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/utils.py", line 193, in raise_on_meta_error
        yield
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/core.py", line 6793, in _emulate
        return func(*_extract_meta(args, True), **_extract_meta(kwargs, True))
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/groupby.py", line 1203, in _groupby_apply_funcs
        r = func(grouped, **func_kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/groupby.py", line 1249, in _apply_func_to_column
        return func(df_like[column])
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/logical/aggregate.py", line 152, in <lambda>
        dd.Aggregation("every", lambda s: s.all(), lambda s0: s0.all())
    AttributeError: 'SeriesGroupBy' object has no attribute 'all'

    The above exception was the direct cause of the following exception:

    Traceback (most recent call last):
      File "/tmp/bug.py", line 21, in <module>
        result2= c.sql("SELECT 1 FROM t0_gpu HAVING EVERY(true)").compute()
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/context.py", line 513, in sql
        return self._compute_table_from_rel(rel, return_futures)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/context.py", line 839, in _compute_table_from_rel
        dc = RelConverter.convert(rel, context=self)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/convert.py", line 61, in convert
        df = plugin_instance.convert(rel, context=context)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/logical/project.py", line 28, in convert
        (dc,) = self.assert_inputs(rel, 1, context)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/base.py", line 84, in assert_inputs
        return [RelConverter.convert(input_rel, context) for input_rel in input_rels]
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/base.py", line 84, in <listcomp>
        return [RelConverter.convert(input_rel, context) for input_rel in input_rels]
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/convert.py", line 61, in convert
        df = plugin_instance.convert(rel, context=context)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/logical/filter.py", line 56, in convert
        (dc,) = self.assert_inputs(rel, 1, context)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/base.py", line 84, in assert_inputs
        return [RelConverter.convert(input_rel, context) for input_rel in input_rels]
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/base.py", line 84, in <listcomp>
        return [RelConverter.convert(input_rel, context) for input_rel in input_rels]
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/convert.py", line 61, in convert
        df = plugin_instance.convert(rel, context=context)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/logical/aggregate.py", line 231, in convert
        df_agg, output_column_order, cc = self._do_aggregations(
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/logical/aggregate.py", line 312, in _do_aggregations
        df_result = self._perform_aggregation(
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/logical/aggregate.py", line 551, in _perform_aggregation
        agg_result = grouped_df.agg(aggregations_dict, **groupby_agg_options)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/groupby.py", line 374, in wrapper
        return func(self, *args, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/groupby.py", line 2884, in agg
        return self.aggregate(
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/nvtx/nvtx.py", line 101, in inner
        result = func(*args, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_cudf/groupby.py", line 218, in aggregate
        return super().aggregate(
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/groupby.py", line 2873, in aggregate
        return super().aggregate(
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/groupby.py", line 2369, in aggregate
        result = aca(
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/core.py", line 6746, in apply_concat_apply
        meta_chunk = _emulate(chunk, *args, udf=True, **chunk_kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/core.py", line 6792, in _emulate
        with raise_on_meta_error(funcname(func), udf=udf), check_numeric_only_deprecation():
      File "/opt/conda/envs/rapids/lib/python3.10/contextlib.py", line 153, in __exit__
        self.gen.throw(typ, value, traceback)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/utils.py", line 214, in raise_on_meta_error
        raise ValueError(msg) from e
    ValueError: Metadata inference failed in `_groupby_apply_funcs`.

    You have supplied a custom function and Dask is unable to 
    determine the type of output that that function returns. 

    To resolve this please provide a meta= keyword.
    The docstring of the Dask function you ran should have more information.

    Original error is below:
    ------------------------
    AttributeError("'SeriesGroupBy' object has no attribute 'all'")

    Traceback:
    ---------
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/utils.py", line 193, in raise_on_meta_error
        yield
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/core.py", line 6793, in _emulate
        return func(*_extract_meta(args, True), **_extract_meta(kwargs, True))
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/groupby.py", line 1203, in _groupby_apply_funcs
        r = func(grouped, **func_kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/groupby.py", line 1249, in _apply_func_to_column
        return func(df_like[column])
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/logical/aggregate.py", line 152, in <lambda>
        dd.Aggregation("every", lambda s: s.all(), lambda s0: s0.all())
    ```



*   ***(6)*** ***GPU Error Bug #6***

    ***URL:*** <https://github.com/dask-contrib/dask-sql/issues/1228#issue-1904373563>

    ***Brief Description:*** * SELECT (\<column> IN (\<string>, \<string>, (CASE \<column> WHEN \<string> THEN \<column> END ), ((\<column>)||(\<string>)))) FROM \<table>  * brings errors,  when using 'GPU'.

    ***Status:*** None

    ***Test Case:***

    ```python
    import pandas as pd
    import dask.dataframe as dd
    from dask_sql import Context

    c = Context()

    df0 = pd.DataFrame({
        'c0': ['A'],
    })
    t0 = dd.from_pandas(df0, npartitions=1)

    c.create_table('t0', t0, gpu=False)
    c.create_table('t0_gpu', t0, gpu=True)

    print('CPU Result:')
    result1= c.sql("SELECT (t0.c0 IN ('A', 'B', (CASE t0.c0 WHEN 'A' THEN t0.c0 END ),  ((t0.c0)||('C')))) FROM t0").compute()
    print(result1)

    print('GPU Result:')
    result2= c.sql("SELECT (t0_gpu.c0 IN ('A', 'B', (CASE t0_gpu.c0 WHEN 'A' THEN t0_gpu.c0 END ), ((t0_gpu.c0)||('C')))) FROM t0_gpu").compute()
    print(result2)
    ```

    ```bash
    INFO:numba.cuda.cudadrv.driver:init
    CPU Result:
       t0.c0 IN (Map { iter: Iter([Utf8("A"), Utf8("B"), CASE t0.c0 WHEN Utf8("A") THEN t0.c0 END, t0.c0 || Utf8("C")]) })
    0                                               True                                                                  
    GPU Result:
    Traceback (most recent call last):
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/column/column.py", line 2210, in as_column
        memoryview(arbitrary), dtype=dtype, nan_as_null=nan_as_null
    TypeError: memoryview: a bytes-like object is required, not 'tuple'

    During handling of the above exception, another exception occurred:

    Traceback (most recent call last):
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/column/column.py", line 2327, in as_column
        pa.array(
      File "pyarrow/array.pxi", line 320, in pyarrow.lib.array
      File "pyarrow/array.pxi", line 39, in pyarrow.lib._sequence_to_array
      File "pyarrow/error.pxi", line 144, in pyarrow.lib.pyarrow_internal_check_status
      File "pyarrow/error.pxi", line 123, in pyarrow.lib.check_status
    pyarrow.lib.ArrowTypeError: Expected bytes, got a 'Series' object

    During handling of the above exception, another exception occurred:

    Traceback (most recent call last):
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/column/column.py", line 2375, in _construct_array
        arbitrary = cupy.asarray(arbitrary, dtype=dtype)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cupy/_creation/from_data.py", line 76, in asarray
        return _core.array(a, dtype, False, order)
      File "cupy/_core/core.pyx", line 2382, in cupy._core.core.array
      File "cupy/_core/core.pyx", line 2406, in cupy._core.core.array
      File "cupy/_core/core.pyx", line 2531, in cupy._core.core._array_default
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/core.py", line 591, in __array__
        x = np.array(self._computed)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/frame.py", line 454, in __array__
        raise TypeError(
    TypeError: Implicit conversion to a host NumPy array via __array__ is not allowed, To explicitly construct a GPU matrix, consider using .to_cupy()
    To explicitly construct a host matrix, consider using .to_numpy().

    During handling of the above exception, another exception occurred:

    Traceback (most recent call last):
      File "/tmp/bug.py", line 20, in <module>
        result2= c.sql("SELECT (t0_gpu.c0 IN ('A', 'B', (CASE t0_gpu.c0 WHEN 'A' THEN t0_gpu.c0 END ), ((t0_gpu.c0)||('C')))) FROM t0_gpu").compute()
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/context.py", line 513, in sql
        return self._compute_table_from_rel(rel, return_futures)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/context.py", line 839, in _compute_table_from_rel
        dc = RelConverter.convert(rel, context=self)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/convert.py", line 61, in convert
        df = plugin_instance.convert(rel, context=context)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/logical/project.py", line 57, in convert
        new_columns[random_name] = RexConverter.convert(
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rex/convert.py", line 74, in convert
        df = plugin_instance.convert(rel, rex, dc, context=context)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rex/core/call.py", line 1129, in convert
        return operation(*operands, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rex/core/call.py", line 77, in __call__
        return self.f(*operands, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rex/core/call.py", line 965, in inList
        result = series.isin(operands)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/core.py", line 4147, in isin
        return super().isin(values)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/core.py", line 3383, in isin
        meta = self._meta_nonempty.isin(values)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/nvtx/nvtx.py", line 101, in inner
        result = func(*args, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/series.py", line 2868, in isin
        {self.name: self._column.isin(values)}, index=self.index
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/column/column.py", line 814, in isin
        lhs, rhs = self._process_values_for_isin(values)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/column/column.py", line 832, in _process_values_for_isin
        rhs = as_column(values, nan_as_null=False)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/column/column.py", line 2360, in as_column
        _construct_array(arbitrary, dtype),
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/column/column.py", line 2381, in _construct_array
        and infer_dtype(arbitrary)
      File "pandas/_libs/lib.pyx", line 1491, in pandas._libs.lib.infer_dtype
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/pandas/core/dtypes/cast.py", line 1784, in construct_1d_object_array_from_listlike
        result[:] = values
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/core.py", line 591, in __array__
        x = np.array(self._computed)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/frame.py", line 454, in __array__
        raise TypeError(
    TypeError: Implicit conversion to a host NumPy array via __array__ is not allowed, To explicitly construct a GPU matrix, consider using .to_cupy()
    To explicitly construct a host matrix, consider using .to_numpy().
    ```



*   ***(7)*** ***GPU Error Bug #7***

    ***URL:***<https://github.com/dask-contrib/dask-sql/issues/1276#issue-2015425041>

    ***Brief Description:*** *SELECT (('b햦]D7Jr31')||((CASE 'Kx}lzJ^' WHEN \<column> THEN '' END ))) FROM \<table>*  brings errors,  when using 'GPU'.

    ***Status:*** None

    ***Test Case:***

    ```python
    import pandas as pd
    import dask.dataframe as dd
    from dask_sql import Context

    c = Context()

    df0 = pd.DataFrame({
        'c0': ['e'],
        'c1': ['o'],
    })
    t0 = dd.from_pandas(df0, npartitions=1)
    c.create_table('t0', t0, gpu=False)
    c.create_table('t0_gpu', t0, gpu=True)

    print('Result1:')
    result1= c.sql("SELECT (('b햦]D7Jr31')||((CASE 'Kx}lzJ^' WHEN t0.c1 THEN '' END ))) FROM t0").compute()
    print(result1)

    print('Result2:')
    result2= c.sql("SELECT (('b햦]D7Jr31')||((CASE 'Kx}lzJ^' WHEN t0_gpu.c1 THEN '' END ))) FROM t0_gpu").compute()
    print(result2)
    ```

    ```bash
    INFO:numba.cuda.cudadrv.driver:init
    Result1:
      Utf8("b햦]D7Jr31") || CASE Utf8("Kx}lzJ^") WHEN t0.c1 THEN Utf8("") END
    0                                                NaN                    
    Result2:
    Traceback (most recent call last):
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/indexed_frame.py", line 2180, in _apply
        kernel, retty = _compile_or_get(
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/nvtx/nvtx.py", line 101, in inner
        result = func(*args, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/udf/utils.py", line 268, in _compile_or_get
        kernel, scalar_return_type = kernel_getter(frame, func, args)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/udf/scalar_function.py", line 55, in _get_scalar_kernel
        scalar_return_type = _get_udf_return_type(sr_type, func, args)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/nvtx/nvtx.py", line 101, in inner
        result = func(*args, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/udf/utils.py", line 88, in _get_udf_return_type
        ptx, output_type = cudautils.compile_udf(func, compile_sig)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/utils/cudautils.py", line 258, in compile_udf
        output_type = numpy_support.as_dtype(return_type).type
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/numba/np/numpy_support.py", line 159, in as_dtype
        raise errors.NumbaNotImplementedError(msg)
    numba.core.errors.NumbaNotImplementedError: unicode_type cannot be represented as a NumPy dtype

    The above exception was the direct cause of the following exception:

    Traceback (most recent call last):
      File "/tmp/bug17/bug17.py", line 20, in <module>
        result2= c.sql("SELECT (('b햦]D7Jr31')||((CASE 'Kx}lzJ^' WHEN t0_gpu.c1 THEN '' END ))) FROM t0_gpu").compute()
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/base.py", line 314, in compute
        (result,) = compute(self, traverse=False, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/base.py", line 599, in compute
        results = schedule(dsk, keys, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/threaded.py", line 89, in get
        results = get_async(
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/local.py", line 511, in get_async
        raise_exception(exc, tb)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/local.py", line 319, in reraise
        raise exc
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/local.py", line 224, in execute_task
        result = _execute_task(task, data)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/core.py", line 119, in _execute_task
        return func(*(_execute_task(a, cache) for a in args))
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/optimization.py", line 990, in __call__
        return core.get(self.dsk, self.outkey, dict(zip(self.inkeys, args)))
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/core.py", line 149, in get
        result = _execute_task(task, cache)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/core.py", line 119, in _execute_task
        return func(*(_execute_task(a, cache) for a in args))
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/core.py", line 119, in <genexpr>
        return func(*(_execute_task(a, cache) for a in args))
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/core.py", line 119, in _execute_task
        return func(*(_execute_task(a, cache) for a in args))
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/core.py", line 119, in <genexpr>
        return func(*(_execute_task(a, cache) for a in args))
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/core.py", line 119, in _execute_task
        return func(*(_execute_task(a, cache) for a in args))
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/utils.py", line 73, in apply
        return func(*args, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/core.py", line 7006, in apply_and_enforce
        df = func(*args, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/utils.py", line 1105, in __call__
        return getattr(__obj, self.method)(*args, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/nvtx/nvtx.py", line 101, in inner
        result = func(*args, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/series.py", line 2473, in apply
        result = self._apply(func, _get_scalar_kernel, *args, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/contextlib.py", line 79, in inner
        return func(*args, **kwds)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/nvtx/nvtx.py", line 101, in inner
        result = func(*args, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/indexed_frame.py", line 2184, in _apply
        raise ValueError(
    ValueError: user defined function compilation failed.
    ```



*   ***(8)*** ***GPU Error Bug #8***

    ***URL:***<https://github.com/dask-contrib/dask-sql/issues/1277#issue-2015426552>

    ***Brief Description: **SELECT (((\<column> LIKE '\뽞^' ESCAPE 'M')) IS NULL) FROM \<tables>* brings errors,  when using 'GPU'.

    ***Status:*** None

    ***Test Case:***

    ```python
    import pandas as pd
    import dask.dataframe as dd
    from dask_sql import Context

    c = Context()

    df0 = pd.DataFrame({
        'c0': ["TIMESTAMP '1970-08-16 10:28:23'"],
        'c1': ["DATE '2002-01-29'"],
        'c2': ["DATE '2021-10-05'"],
        'c3': [True],
    })
    t0 = dd.from_pandas(df0, npartitions=1)
    c.create_table('t0', t0, gpu=False)
    c.create_table('t0_gpu', t0, gpu=True)

    df1 = pd.DataFrame({
        'c0': ['b'],
        'c1': ["TIMESTAMP '1972-02-28 05:27:02'"],
        'c2': [836.0000],
        'c3': ['CAST((-106) AS TINYINT)'],
    })
    t1 = dd.from_pandas(df1, npartitions=1)
    c.create_table('t1', t1, gpu=False)
    c.create_table('t1_gpu', t1, gpu=True)

    print('Result1:')
    result1= c.sql("SELECT (((t1.c1 LIKE '\뽞^' ESCAPE 'M')) IS NULL) FROM t0, t1").compute()
    print(result1)

    print('Result2:')
    result2= c.sql("SELECT (((t1_gpu.c1 LIKE '\뽞^' ESCAPE 'M')) IS NULL) FROM t0_gpu, t1_gpu").compute()
    print(result2)
    ```

    ```bash
    INFO:numba.cuda.cudadrv.driver:init
    Result1:
       t1.c1 LIKE Utf8("\뽞^") CHAR 'M' IS NULL
    0                                    False
    Result2:
    Traceback (most recent call last):
      File "/tmp/bug18/bug18.py", line 32, in <module>
        result2= c.sql("SELECT (((t1_gpu.c1 LIKE '\뽞^' ESCAPE 'M')) IS NULL) FROM t0_gpu, t1_gpu").compute()
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/context.py", line 513, in sql
        return self._compute_table_from_rel(rel, return_futures)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/context.py", line 839, in _compute_table_from_rel
        dc = RelConverter.convert(rel, context=self)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/convert.py", line 61, in convert
        df = plugin_instance.convert(rel, context=context)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rel/logical/project.py", line 57, in convert
        new_columns[random_name] = RexConverter.convert(
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rex/convert.py", line 74, in convert
        df = plugin_instance.convert(rel, rex, dc, context=context)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rex/core/call.py", line 1101, in convert
        operands = [
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rex/core/call.py", line 1102, in <listcomp>
        RexConverter.convert(rel, o, dc, context=context)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rex/convert.py", line 74, in convert
        df = plugin_instance.convert(rel, rex, dc, context=context)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rex/core/call.py", line 1129, in convert
        return operation(*operands, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rex/core/call.py", line 77, in __call__
        return self.f(*operands, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask_sql/physical/rex/core/call.py", line 443, in regex
        return test.str.match(transformed_regex, flags=flags).astype("boolean")
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/accessor.py", line 13, in func
        return self._function_map(attr, *args, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/accessor.py", line 106, in _function_map
        meta = self._delegate_method(
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/dask/dataframe/accessor.py", line 92, in _delegate_method
        out = getattr(getattr(obj, accessor, obj), attr)(*args, **kwargs)
      File "/opt/conda/envs/rapids/lib/python3.10/site-packages/cudf/core/column/string.py", line 4291, in match
        libstrings.match_re(self._column, pat, flags)
      File "/opt/conda/envs/rapids/lib/python3.10/contextlib.py", line 79, in inner
        return func(*args, **kwds)
      File "contains.pyx", line 87, in cudf._lib.strings.contains.match_re
    RuntimeError: CUDF failure at:/rapids/cudf/cpp/src/strings/regex/regcomp.cpp:521: invalid regex pattern: bad escape character at position 2
    ```



#### #3 Logic Bug

*   ***(1)*** ***GPU Error Bug #1***

    ***URL:*** <https://github.com/dask-contrib/dask-sql/issues/1226#issue-1903129160>

    ***Brief Description:***  *SELECT (\<string>)||(\<column(decimal)>) FROM \<table>* brings different results, when using CPU and GPU.

    ***Status:*** Confirmed

    ***Test Case:***

    ```python
    import pandas as pd
    import dask.dataframe as dd
    from dask_sql import Context

    c = Context()

    df = pd.DataFrame({
        'c0': [0.5113391810437729]
    })
    t1 = dd.from_pandas(df, npartitions=1)

    c.create_table('t1', t1, gpu=False)
    c.create_table('t1_gpu', t1, gpu=True)

    print('CPU Result:')
    result1= c.sql("SELECT ('A')||(t1.c0) FROM t1").compute()
    print(result1)

    print('GPU Result:')
    result2= c.sql("SELECT ('A')||(t1_gpu.c0) FROM t1_gpu").compute()
    print(result2)
    ```

    ```bash
    CPU Result:
        Utf8("A") || t1.c0
    0  A0.5113391810437729
    GPU Result:
      Utf8("A") || t1_gpu.c0
    0           A0.511339181
    ```



*   ***(2)*** ***GPU Error Bug #2***

    ***URL:*** <https://github.com/dask-contrib/dask-sql/issues/1236#issue-1905145952>

    ***Brief Description:*** *SELECT (CASE (\<column>) WHEN \<number> THEN \<number> END) FROM \<table>* brings different results, when using CPU and GPU.

    ***Status:*** None

    ***Test Case:***

    ```python
    import pandas as pd
    import dask.dataframe as dd
    from dask_sql import Context

    c = Context()

    df0 = pd.DataFrame({
        'c0': [0.898116830962538],
    })
    t0 = dd.from_pandas(df0, npartitions=1)

    c.create_table('t0', t0, gpu=False)
    c.create_table('t0_gpu', t0, gpu=True)

    print('CPU Result:')
    result1= c.sql("SELECT (CASE (t0.c0) WHEN 1 THEN 1 END) FROM t0").compute()
    print(result1)

    print('GPU Result:')
    result2= c.sql("SELECT (CASE (t0_gpu.c0) WHEN 1 THEN 1 END) FROM t0_gpu").compute()
    print(result2)
    ```

    ```bash
    INFO:numba.cuda.cudadrv.driver:init
    CPU Result:
       CASE t0.c0 WHEN Int64(1) THEN Int64(1) END
    0                                         NaN
    GPU Result:
      CASE t0_gpu.c0 WHEN Int64(1) THEN Int64(1) END
    0                                           <NA>
    INFO:numba.cuda.cudadrv.driver:add pending dealloc: module_unload ? bytes
    ```



*   ***(3)*** ***GPU Error Bug #3***

    ***URL:*** <https://github.com/dask-contrib/dask-sql/issues/1235#issue-1905046307>

    ***Brief Description:*** *SELECT ( (CASE (CASE (\<number>) WHEN \<column> THEN (\<number>) END ) WHEN \<number> THEN (\<number>) ELSE \<column> END )) FROM \<table>* brings different results, when using CPU and GPU.

    ***Status:*** None

    ***Test Case:***

    ```python
    import pandas as pd
    import dask.dataframe as dd
    from dask_sql import Context

    c = Context()

    df0 = pd.DataFrame({
        'c1': [0.6926717947094722],
        'c2': [True],
    })
    t0 = dd.from_pandas(df0, npartitions=1)

    c.create_table('t0', t0, gpu=False)
    c.create_table('t0_gpu', t0, gpu=True)

    print('CPU Result:')
    result1= c.sql("SELECT ( (CASE (CASE (1) WHEN t0.c1 THEN (1.0) END ) WHEN 0.1 THEN (47) ELSE t0.c1 END )) FROM t0").compute()
    print(result1)

    print('GPU Result:')
    result2= c.sql("SELECT ( (CASE (CASE (1) WHEN t0_gpu.c1 THEN (1.0) END ) WHEN 0.1 THEN (47) ELSE t0_gpu.c1 END )) FROM t0_gpu").compute()
    print(result2)
    ```

    ```bash
    INFO:numba.cuda.cudadrv.driver:init
    Result1:
       CASE CASE Int64(1) WHEN t0.c1 THEN Float64(1) END WHEN Float64(0.1) THEN Int64(47) ELSE t0.c1 END
    0                                           0.692672                                                
    Result2:
       CASE CASE Int64(1) WHEN t0_gpu.c1 THEN Float64(1) END WHEN Float64(0.1) THEN Int64(47) ELSE t0_gpu.c1 END
    0                                               47.0                                                        
    INFO:numba.cuda.cudadrv.driver:add pending dealloc: module_unload ? bytes
    ```

<!---->

*   ***(4)*** ***GPU Error Bug #4***

    ***URL:*** <https://github.com/dask-contrib/dask-sql/issues/1234#issue-1905003214>

    ***Brief Description:*** *SELECT ((\<column>) IS DISTINCT FROM ((CASE \<column> WHEN \<number> THEN \<number> END ))) FROM \<table>* brings different results, when using CPU and GPU.

    ***Status:*** None

    ***Test Case:***

    ```python
    import pandas as pd
    import dask.dataframe as dd
    from dask_sql import Context

    c = Context()

    df0 = pd.DataFrame({
        'c0': [39.0000],
    })
    t0 = dd.from_pandas(df0, npartitions=1)

    c.create_table('t0', t0, gpu=False)
    c.create_table('t0_gpu', t0, gpu=True)

    print('CPU Result:')
    result1= c.sql("SELECT ((t0.c0) IS DISTINCT FROM ((CASE t0.c0 WHEN 0.1 THEN (1) END ))) FROM t0").compute()
    print(result1)

    print('GPU Result:')
    result2= c.sql("SELECT ((t0_gpu.c0) IS DISTINCT FROM ((CASE t0_gpu.c0 WHEN 0.1 THEN (1) END ))) FROM t0_gpu").compute()
    print(result2)
    ```

    ```bash
    INFO:numba.cuda.cudadrv.driver:init
    CPU Result:
       t0.c0 IS DISTINCT FROM CASE t0.c0 WHEN Float64(0.1) THEN Int64(1) END
    0                                               True                    
    GPU Result:
      t0_gpu.c0 IS DISTINCT FROM CASE t0_gpu.c0 WHEN Float64(0.1) THEN Int64(1) END
    0                                               <NA>                           
    INFO:numba.cuda.cudadrv.driver:add pending dealloc: module_unload ? bytes
    ```

<!---->

*   ***(5)*** ***GPU Error Bug #5***

    ***URL:*** <https://github.com/dask-contrib/dask-sql/issues/1233#issue-1904946380>

    ***Brief Description:*** *SELECT (\<number> NOT BETWEEN (CASE ((\<column>)) WHEN (1) THEN 0 END ) AND (\<column>)) FROM \<table> * brings different results, when using CPU and GPU.

    ***Status:*** None

    ***Test Case:***

    ```python
    import pandas as pd
    import dask.dataframe as dd
    from dask_sql import Context

    c = Context()

    df0 = pd.DataFrame({
        'c0': [True],
        'c1': [3666.0000],
        'c2': [0.3820597044277436],
    })
    t0 = dd.from_pandas(df0, npartitions=1)

    c.create_table('t0', t0, gpu=False)
    c.create_table('t0_gpu', t0, gpu=True)

    print('CPU Result:')
    result1= c.sql("SELECT ((1) NOT BETWEEN (CASE ((t0.c1)) WHEN (1) THEN 0 END ) AND (t0.c0)) FROM t0").compute()
    print(result1)

    print('GPU Result:')
    result2= c.sql("SELECT ((1) NOT BETWEEN (CASE ((t0_gpu.c1)) WHEN (1) THEN 0 END ) AND (t0_gpu.c0)) FROM t0_gpu").compute()
    print(result2)
    ```

    ```bash
    INFO:numba.cuda.cudadrv.driver:init
    CPU Result:
       Int64(1) NOT BETWEEN CASE t0.c1 WHEN Int64(1) THEN Int64(0) END AND t0.c0
    0                                              False                        
    GPU Result:
      Int64(1) NOT BETWEEN CASE t0_gpu.c1 WHEN Int64(1) THEN Int64(0) END AND t0_gpu.c0
    0                                               <NA>                               
    INFO:numba.cuda.cudadrv.driver:add pending dealloc: module_unload ? bytes
    ```