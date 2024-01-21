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

*   ***(1)***  ***Crash Bug #1 ***

    ***URL:***  <https://github.com/heavyai/heavydb/issues/794#issue-1875017288>

    ***Brief Description：***Create table with ( special \<property> = value ) causes crash, after set EXECUTOR\_DEVICE='GPU'.

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

*   ***(2) Crash Bug #2 ***

    ***URL:*** <https://github.com/heavyai/heavydb/issues/795>

    ***Brief Description：***Insert values after create table with ( page\_size = value (too small or large) ) causes crash, after set EXECUTOR\_DEVICE='GPU'.

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

    ***Test Case: ***

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

    ***Brief Description: **\<column> NOT IN \<column(overflow)> *in WHERE condition brings errors,  when set EXECUTOR\_DEVICE 'GPU'.

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

    ***Brief Description: *** *SELECT \* FROM \<table> WHERE ((\<column> + \<column>) < \<column>) OR (\<column> = \<column>)* brings errors,  when set EXECUTOR\_DEVICE 'GPU'.

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

    ***Brief Description:*** * CAST(\<column>+\<column>(overflow) AS BOOLEAN)*  in WHERE condition brings errors,  when set EXECUTOR\_DEVICE 'GPU'.

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

    ***Brief Description:*** *SELECT DISTINCT \<column> FROM \<table> ORDER BY 1 DESC LIMIT 10 * brings different results , when set EXECUTOR\_DEVICE 'CPU' and 'GPU'.

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

*   (1)

#### &#x20;#2 Error Bug

#### &#x20;#3 Logic Bug
