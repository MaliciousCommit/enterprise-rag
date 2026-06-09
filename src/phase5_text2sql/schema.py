# src/phase5_text2sql/schema.py
#
# PostgreSQL schema for the Kubernetes IT Operations database.
#
# WHY THESE 7 TABLES?
# Each table answers a category of operational questions:
#
#   clusters    → "Which clusters are running K8s 1.29?"
#   pods        → "How many pods are in CrashLoopBackOff right now?"
#   nodes       → "Which nodes have memory pressure?"
#   incidents   → "What P1 incidents happened this month?"
#   deployments → "What was deployed to prod this week?"
#   alerts      → "How many critical alerts fired in the last hour?"
#   audit_log   → "Who asked what, when?" (compliance + debugging)
#
# WHY SQL RATHER THAN QDRANT FOR THIS DATA?
# This data is STRUCTURED and RELATIONAL:
#   - Pod counts, aggregations, GROUP BY → SQL strength
#   - Exact matches: status = 'CrashLoopBackOff' → SQL strength
#   - Joins across tables (pods JOIN clusters) → SQL strength
#
# Vector search would be WRONG here:
#   - "How many pods are failing?" is not a semantic similarity problem
#   - You need COUNT(*) not approximate nearest neighbours
#   - Real-time cluster state changes constantly — you need live data, not embeddings
#
# SCHEMA DESIGN DECISIONS:
#   recorded_at: separate from created_at — allows time-series queries
#              ("pods in error state in the last 30 minutes")
#   JSONB labels/annotations: Kubernetes labels are arbitrary key-value pairs
#   TEXT[]: affected_services uses PostgreSQL array type for multi-value columns

# ── DDL ──────────────────────────────────────────────────────────────────────

CREATE_TABLES_SQL = """

-- Table 1: Cluster registry
CREATE TABLE IF NOT EXISTS clusters (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    cluster_name    VARCHAR(255) NOT NULL UNIQUE,
    k8s_version     VARCHAR(50)  NOT NULL,
    environment     VARCHAR(50)  NOT NULL CHECK (environment IN ('prod','staging','dev')),
    region          VARCHAR(100),
    cloud_provider  VARCHAR(50),
    node_count      INTEGER      DEFAULT 0,
    status          VARCHAR(50)  DEFAULT 'active',
    created_at      TIMESTAMPTZ  DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  DEFAULT NOW()
);

-- Table 2: Pod state snapshots (updated by metrics scraper every 30s)
CREATE TABLE IF NOT EXISTS pods (
    id                     UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    cluster_id             UUID        REFERENCES clusters(id) ON DELETE CASCADE,
    pod_name               VARCHAR(255) NOT NULL,
    namespace              VARCHAR(255) NOT NULL,
    node_name              VARCHAR(255),
    status                 VARCHAR(100),
    phase                  VARCHAR(50),
    restart_count          INTEGER     DEFAULT 0,
    cpu_request_millicores INTEGER,
    cpu_limit_millicores   INTEGER,
    memory_request_mi      INTEGER,
    memory_limit_mi        INTEGER,
    cpu_usage_millicores   FLOAT,
    memory_usage_mi        FLOAT,
    image                  VARCHAR(500),
    labels                 JSONB       DEFAULT '{}',
    recorded_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pods_cluster_status
    ON pods(cluster_id, status, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_pods_namespace
    ON pods(cluster_id, namespace);

-- Table 3: Node health and capacity
CREATE TABLE IF NOT EXISTS nodes (
    id                         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    cluster_id                 UUID        REFERENCES clusters(id),
    node_name                  VARCHAR(255) NOT NULL,
    status                     VARCHAR(50),
    cpu_capacity_millicores    INTEGER,
    memory_capacity_mi         INTEGER,
    cpu_allocatable_millicores INTEGER,
    memory_allocatable_mi      INTEGER,
    cpu_usage_millicores       FLOAT,
    memory_usage_mi            FLOAT,
    disk_pressure              BOOLEAN     DEFAULT FALSE,
    memory_pressure            BOOLEAN     DEFAULT FALSE,
    pid_pressure               BOOLEAN     DEFAULT FALSE,
    k8s_version                VARCHAR(50),
    recorded_at                TIMESTAMPTZ DEFAULT NOW()
);

-- Table 4: Incident history
CREATE TABLE IF NOT EXISTS incidents (
    id                UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    incident_id       VARCHAR(100) UNIQUE NOT NULL,
    title             VARCHAR(500) NOT NULL,
    description       TEXT,
    severity          VARCHAR(20)  NOT NULL CHECK (severity IN ('P1','P2','P3','P4')),
    status            VARCHAR(50)  DEFAULT 'open',
    cluster_id        UUID         REFERENCES clusters(id),
    namespace         VARCHAR(255),
    affected_services TEXT[],
    root_cause        TEXT,
    resolution        TEXT,
    created_at        TIMESTAMPTZ  DEFAULT NOW(),
    resolved_at       TIMESTAMPTZ,
    created_by        VARCHAR(255)
);

CREATE INDEX IF NOT EXISTS idx_incidents_severity_status
    ON incidents(severity, status, created_at DESC);

-- Table 5: Deployment history
CREATE TABLE IF NOT EXISTS deployments (
    id                 UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    cluster_id         UUID        REFERENCES clusters(id),
    deployment_name    VARCHAR(255) NOT NULL,
    namespace          VARCHAR(255) NOT NULL,
    image              VARCHAR(500) NOT NULL,
    image_tag          VARCHAR(200),
    desired_replicas   INTEGER,
    available_replicas INTEGER,
    ready_replicas     INTEGER,
    strategy           VARCHAR(50),
    deployed_by        VARCHAR(255),
    git_commit_sha     VARCHAR(64),
    deployed_at        TIMESTAMPTZ DEFAULT NOW(),
    status             VARCHAR(50)
);

CREATE INDEX IF NOT EXISTS idx_deployments_cluster_deployed
    ON deployments(cluster_id, deployed_at DESC);

-- Table 6: Alert history from Prometheus/Alertmanager
CREATE TABLE IF NOT EXISTS alerts (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    cluster_id   UUID        REFERENCES clusters(id),
    alert_name   VARCHAR(255) NOT NULL,
    severity     VARCHAR(20),
    namespace    VARCHAR(255),
    pod_name     VARCHAR(255),
    node_name    VARCHAR(255),
    description  TEXT,
    labels       JSONB       DEFAULT '{}',
    status       VARCHAR(20) DEFAULT 'firing',
    fired_at     TIMESTAMPTZ DEFAULT NOW(),
    resolved_at  TIMESTAMPTZ,
    incident_id  UUID        REFERENCES incidents(id)
);

CREATE INDEX IF NOT EXISTS idx_alerts_cluster_severity
    ON alerts(cluster_id, severity, fired_at DESC);

-- Table 7: Audit log — every RAG/SQL query logged for compliance
CREATE TABLE IF NOT EXISTS audit_log (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id        VARCHAR(255) NOT NULL,
    user_id           VARCHAR(255) NOT NULL,
    question          TEXT         NOT NULL,
    intent            VARCHAR(50),
    answer            TEXT,
    sql_query         TEXT,
    sql_approved      BOOLEAN,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    latency_ms        FLOAT,
    cache_hit         BOOLEAN     DEFAULT FALSE,
    created_at        TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_log_user_created
    ON audit_log(user_id, created_at DESC);
"""

# Schema description injected into the SQL generation prompt.
# This is what the LLM reads to understand the database structure.
# CRITICAL: must be accurate — wrong column names → SQL errors.
SCHEMA_DESCRIPTION = """
DATABASE: enterprise_rag (PostgreSQL 16)
PURPOSE:  Kubernetes IT Operations — live cluster state and incident history

TABLE: clusters
  id              UUID PRIMARY KEY
  cluster_name    VARCHAR  -- e.g. 'prod-us-east-1', 'staging-eu-west-1'
  k8s_version     VARCHAR  -- e.g. '1.29.3', '1.28.8'
  environment     VARCHAR  -- 'prod', 'staging', 'dev'
  region          VARCHAR  -- cloud region
  cloud_provider  VARCHAR  -- 'aws', 'gcp', 'azure', 'on-prem'
  node_count      INTEGER
  status          VARCHAR  -- 'active', 'maintenance', 'decommissioned'
  created_at, updated_at  TIMESTAMPTZ

TABLE: pods
  id              UUID PRIMARY KEY
  cluster_id      UUID REFERENCES clusters(id)
  pod_name        VARCHAR  -- e.g. 'payment-svc-7d9f8b-xk2p'
  namespace       VARCHAR  -- e.g. 'prod', 'kube-system', 'monitoring'
  node_name       VARCHAR
  status          VARCHAR  -- 'Running', 'CrashLoopBackOff', 'OOMKilled',
                           --  'Pending', 'Error', 'Completed', 'Terminating'
  phase           VARCHAR  -- 'Running', 'Pending', 'Failed', 'Succeeded'
  restart_count   INTEGER  -- total container restarts
  cpu_request_millicores, cpu_limit_millicores   INTEGER  -- 1000 = 1 CPU core
  memory_request_mi, memory_limit_mi             INTEGER  -- mebibytes
  cpu_usage_millicores, memory_usage_mi          FLOAT    -- live metrics
  image           VARCHAR
  labels          JSONB    -- Kubernetes labels as JSON object
  recorded_at     TIMESTAMPTZ  -- when this snapshot was taken

TABLE: nodes
  id              UUID PRIMARY KEY
  cluster_id      UUID REFERENCES clusters(id)
  node_name       VARCHAR
  status          VARCHAR  -- 'Ready', 'NotReady', 'SchedulingDisabled'
  cpu_capacity_millicores, cpu_allocatable_millicores   INTEGER
  memory_capacity_mi, memory_allocatable_mi             INTEGER
  cpu_usage_millicores, memory_usage_mi                 FLOAT
  disk_pressure, memory_pressure, pid_pressure          BOOLEAN
  k8s_version     VARCHAR
  recorded_at     TIMESTAMPTZ

TABLE: incidents
  id              UUID PRIMARY KEY
  incident_id     VARCHAR  -- human-readable e.g. 'INC-2024-0042'
  title           VARCHAR
  description     TEXT
  severity        VARCHAR  -- 'P1', 'P2', 'P3', 'P4'
  status          VARCHAR  -- 'open', 'investigating', 'resolved', 'closed'
  cluster_id      UUID REFERENCES clusters(id)
  namespace       VARCHAR
  affected_services TEXT[] -- array of service names
  root_cause, resolution  TEXT
  created_at      TIMESTAMPTZ
  resolved_at     TIMESTAMPTZ  -- NULL if still open
  created_by      VARCHAR

TABLE: deployments
  id              UUID PRIMARY KEY
  cluster_id      UUID REFERENCES clusters(id)
  deployment_name VARCHAR
  namespace       VARCHAR
  image, image_tag VARCHAR
  desired_replicas, available_replicas, ready_replicas  INTEGER
  strategy        VARCHAR  -- 'RollingUpdate', 'Recreate'
  deployed_by     VARCHAR
  git_commit_sha  VARCHAR
  deployed_at     TIMESTAMPTZ
  status          VARCHAR  -- 'success', 'failed', 'in-progress', 'rolled-back'

TABLE: alerts
  id              UUID PRIMARY KEY
  cluster_id      UUID REFERENCES clusters(id)
  alert_name      VARCHAR  -- e.g. 'KubePodCrashLooping', 'NodeMemoryPressure'
  severity        VARCHAR  -- 'critical', 'warning', 'info'
  namespace, pod_name, node_name  VARCHAR
  description     TEXT
  labels          JSONB
  status          VARCHAR  -- 'firing', 'resolved'
  fired_at        TIMESTAMPTZ
  resolved_at     TIMESTAMPTZ  -- NULL if still firing
  incident_id     UUID REFERENCES incidents(id)  -- NULL if not escalated

TABLE: audit_log
  id              UUID PRIMARY KEY
  session_id      VARCHAR
  user_id         VARCHAR
  question        TEXT
  intent          VARCHAR  -- 'rag', 'sql', 'hybrid'
  answer          TEXT
  sql_query       TEXT     -- the generated SQL if applicable
  sql_approved    BOOLEAN  -- whether human approved the SQL
  prompt_tokens, completion_tokens  INTEGER
  latency_ms      FLOAT
  cache_hit       BOOLEAN
  created_at      TIMESTAMPTZ
""".strip()


# ── Sample data ────────────────────────────────────────────────────────────────
# Realistic K8s operational snapshot for testing Text2SQL queries.

SAMPLE_DATA_SQL = """
-- Clusters
INSERT INTO clusters (cluster_name, k8s_version, environment, region, cloud_provider, node_count, status) VALUES
    ('prod-us-east-1',    '1.29.3', 'prod',    'us-east-1',  'aws',   12, 'active'),
    ('staging-eu-west-1', '1.29.1', 'staging', 'eu-west-1',  'aws',    4, 'active'),
    ('dev-local',         '1.28.8', 'dev',     'local',      'on-prem',2, 'active')
ON CONFLICT (cluster_name) DO NOTHING;

-- Nodes (prod cluster)
WITH prod_id AS (SELECT id FROM clusters WHERE cluster_name = 'prod-us-east-1')
INSERT INTO nodes
    (cluster_id, node_name, status, cpu_capacity_millicores, memory_capacity_mi,
     cpu_allocatable_millicores, memory_allocatable_mi, cpu_usage_millicores,
     memory_usage_mi, disk_pressure, memory_pressure, pid_pressure, k8s_version)
SELECT prod_id.id, n.node_name, n.status, n.cpu_cap, n.mem_cap,
       n.cpu_alloc, n.mem_alloc, n.cpu_use, n.mem_use,
       n.disk_p, n.mem_p, n.pid_p, '1.29.3'
FROM prod_id,
(VALUES
    ('prod-node-01', 'Ready',    8000, 16384, 7800, 15000, 4200, 8192,  false, false, false),
    ('prod-node-02', 'Ready',    8000, 16384, 7800, 15000, 6100, 12000, false, true,  false),
    ('prod-node-03', 'Ready',    8000, 32768, 7800, 31000, 3800, 9000,  false, false, false),
    ('prod-node-04', 'NotReady', 8000, 16384, 7800, 15000, 0,    0,     true,  false, false)
) AS n(node_name, status, cpu_cap, mem_cap, cpu_alloc, mem_alloc, cpu_use, mem_use, disk_p, mem_p, pid_p);

-- Pods (realistic mix of statuses)
WITH prod_id AS (SELECT id FROM clusters WHERE cluster_name = 'prod-us-east-1'),
     staging_id AS (SELECT id FROM clusters WHERE cluster_name = 'staging-eu-west-1')
INSERT INTO pods
    (cluster_id, pod_name, namespace, node_name, status, phase,
     restart_count, cpu_limit_millicores, memory_limit_mi, cpu_usage_millicores,
     memory_usage_mi, image)
SELECT c.cluster_id, p.pod_name, p.namespace, p.node_name, p.status, p.phase,
       p.restart_count, p.cpu_limit, p.mem_limit, p.cpu_use, p.mem_use, p.image
FROM (VALUES
    -- Healthy pods
    ('prod', 'api-gateway-6d8f9-x2k4p',     'prod', 'prod-node-01', 'Running',         'Running', 0,  1000, 512,  450, 280, 'api-gateway:2.1.4'),
    ('prod', 'auth-service-7b4c8-m9p2q',    'prod', 'prod-node-01', 'Running',         'Running', 2,  500,  256,  120, 180, 'auth-service:1.8.2'),
    ('prod', 'inventory-svc-5d7f2-p3k9m',   'prod', 'prod-node-02', 'Running',         'Running', 0,  2000, 1024, 890, 640, 'inventory-svc:3.2.1'),
    ('prod', 'notification-9c4f1-r7k2n',    'prod', 'prod-node-03', 'Running',         'Running', 1,  500,  256,  85,  142, 'notification-svc:1.2.0'),
    ('prod', 'redis-cache-0',               'prod', 'prod-node-03', 'Running',         'Running', 0,  500,  512,  45,  320, 'redis:7.2'),
    -- Failing pods
    ('prod', 'payment-svc-7d9f8-xk2p',     'prod', 'prod-node-02', 'CrashLoopBackOff','Running', 14, 500,  256,  0,   0,   'payment-svc:4.1.2'),
    ('prod', 'payment-svc-7d9f8-mk8l',     'prod', 'prod-node-01', 'CrashLoopBackOff','Running', 8,  500,  256,  0,   0,   'payment-svc:4.1.2'),
    ('prod', 'order-processor-6f4d2-p9k3',  'prod', 'prod-node-02', 'OOMKilled',       'Failed',  22, 512,  256,  0,   0,   'order-processor:2.0.1'),
    ('prod', 'ml-inference-8d2c4-t5m1k',   'ml',   'prod-node-03', 'OOMKilled',       'Failed',  5,  4000, 8192, 0,   0,   'ml-inference:1.1.0'),
    ('prod', 'data-pipeline-3b8f1-n4p2q',  'data', 'prod-node-01', 'Error',           'Failed',  3,  2000, 2048, 0,   0,   'data-pipeline:0.9.8'),
    ('prod', 'search-svc-4c9d5-h7k3m',     'prod', 'prod-node-04', 'Pending',         'Pending', 0,  1000, 512,  0,   0,   'search-svc:2.3.0'),
    -- Staging pods
    ('staging', 'api-gateway-6d8f9-s1t2u',   'prod', NULL, 'Running',         'Running', 0, 1000, 512, 200, 140, 'api-gateway:2.1.5-rc1'),
    ('staging', 'payment-svc-test-4k9f2-v3w','test', NULL, 'CrashLoopBackOff','Running', 6, 500,  256, 0,   0,   'payment-svc:4.1.3-beta')
) AS p(env, pod_name, namespace, node_name, status, phase, restart_count, cpu_limit, mem_limit, cpu_use, mem_use, image)
JOIN LATERAL (
    SELECT CASE p.env
        WHEN 'prod'    THEN (SELECT id FROM clusters WHERE cluster_name = 'prod-us-east-1')
        WHEN 'staging' THEN (SELECT id FROM clusters WHERE cluster_name = 'staging-eu-west-1')
    END AS cluster_id
) c ON true;

-- Incidents
WITH prod_id AS (SELECT id FROM clusters WHERE cluster_name = 'prod-us-east-1')
INSERT INTO incidents
    (incident_id, title, severity, status, cluster_id, namespace,
     affected_services, root_cause, resolution, created_by)
SELECT i.incident_id, i.title, i.severity, i.status, prod_id.id,
       i.namespace, i.affected_services::TEXT[], i.root_cause, i.resolution, 'sre-team'
FROM prod_id,
(VALUES
    ('INC-2024-0042', 'payment-svc CrashLoopBackOff in prod', 'P1', 'investigating',
     'prod', '{payment-svc}', 'Container exits with code 1 after OOM in dependency',
     NULL),
    ('INC-2024-0041', 'order-processor OOMKilled repeatedly', 'P2', 'resolved',
     'prod', '{order-processor,inventory-svc}',
     'Memory limit set to 256Mi but heap grew to 512Mi under load',
     'Increased memory limit to 768Mi, added JVM heap config'),
    ('INC-2024-0040', 'prod-node-04 NotReady — disk pressure', 'P2', 'investigating',
     'prod', '{search-svc,analytics-svc}',
     'Log rotation failed, disk filled to 98%', NULL),
    ('INC-2024-0038', 'API gateway latency spike', 'P3', 'resolved',
     'prod', '{api-gateway}',
     'Redis connection pool exhausted during traffic spike',
     'Increased connection pool limit, added circuit breaker'),
    ('INC-2024-0035', 'Staging deploy pipeline broken', 'P3', 'closed',
     'test', '{ci-cd}',
     'Docker registry credentials expired', 'Rotated credentials')
) AS i(incident_id, title, severity, status, namespace, affected_services, root_cause, resolution)
ON CONFLICT (incident_id) DO NOTHING;

-- Deployments (last 7 days)
WITH prod_id AS (SELECT id FROM clusters WHERE cluster_name = 'prod-us-east-1'),
     staging_id AS (SELECT id FROM clusters WHERE cluster_name = 'staging-eu-west-1')
INSERT INTO deployments
    (cluster_id, deployment_name, namespace, image, image_tag,
     desired_replicas, available_replicas, ready_replicas,
     strategy, deployed_by, status, deployed_at)
VALUES
    ((SELECT id FROM clusters WHERE cluster_name='prod-us-east-1'),
     'api-gateway', 'prod', 'api-gateway', '2.1.4', 3, 3, 3,
     'RollingUpdate', 'alice@company.com', 'success', NOW() - INTERVAL '2 hours'),
    ((SELECT id FROM clusters WHERE cluster_name='prod-us-east-1'),
     'payment-svc', 'prod', 'payment-svc', '4.1.2', 3, 0, 0,
     'RollingUpdate', 'bob@company.com', 'failed', NOW() - INTERVAL '3 hours'),
    ((SELECT id FROM clusters WHERE cluster_name='prod-us-east-1'),
     'order-processor', 'prod', 'order-processor', '2.0.1', 2, 0, 0,
     'RollingUpdate', 'carol@company.com', 'failed', NOW() - INTERVAL '6 hours'),
    ((SELECT id FROM clusters WHERE cluster_name='staging-eu-west-1'),
     'api-gateway', 'prod', 'api-gateway', '2.1.5-rc1', 1, 1, 1,
     'Recreate', 'alice@company.com', 'success', NOW() - INTERVAL '1 hour');

-- Alerts
WITH prod_id AS (SELECT id FROM clusters WHERE cluster_name = 'prod-us-east-1')
INSERT INTO alerts
    (cluster_id, alert_name, severity, namespace, pod_name, description, status, fired_at)
SELECT prod_id.id, a.alert_name, a.severity, a.namespace, a.pod_name, a.description,
       a.status, NOW() - a.ago::INTERVAL
FROM prod_id,
(VALUES
    ('KubePodCrashLooping',   'critical', 'prod', 'payment-svc-7d9f8-xk2p',
     'Pod has been restarting for > 15 minutes',       'firing',   '3 hours'),
    ('KubePodCrashLooping',   'critical', 'prod', 'payment-svc-7d9f8-mk8l',
     'Pod has been restarting for > 15 minutes',       'firing',   '3 hours'),
    ('KubePodOOMKilled',      'critical', 'prod', 'order-processor-6f4d2-p9k3',
     'Container killed by OOM killer',                 'firing',   '6 hours'),
    ('KubeNodeNotReady',      'critical', 'prod', NULL,
     'Node prod-node-04 not ready for > 5 minutes',   'firing',   '1 hour'),
    ('KubePodOOMKilled',      'warning',  'ml',   'ml-inference-8d2c4-t5m1k',
     'ML inference container exceeded memory limit',   'firing',   '2 hours'),
    ('KubeDeploymentRollout', 'warning',  'prod', NULL,
     'payment-svc rollout has not completed in 10min', 'firing',   '3 hours'),
    ('NodeMemoryPressure',    'warning',  NULL,   NULL,
     'Node prod-node-02 memory usage > 85%',           'firing',   '30 minutes'),
    ('KubePodNotReady',       'warning',  'prod', 'search-svc-4c9d5-h7k3m',
     'Pod not ready: node is NotReady',                'firing',   '1 hour')
) AS a(alert_name, severity, namespace, pod_name, description, status, ago);
"""
