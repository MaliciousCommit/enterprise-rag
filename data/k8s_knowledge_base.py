# data/k8s_knowledge_base.py
#
# The Kubernetes IT Operations knowledge base for Module 1.
# 10 documents covering the most common K8s operational topics.
#
# Each document is a dict:
#   document_id: unique identifier (used for deduplication in Phase 6)
#   source:      human-readable path (shown in citations)
#   category:    document type (runbook, guide, reference)
#   k8s_version: which K8s version this applies to
#   content:     the full document text (will be chunked by ingestion.py)
#
# In production, these come from:
#   - Git repositories (markdown files)
#   - Confluence (via REST API)
#   - PagerDuty postmortems (via REST API)
#   - S3 buckets (PDF runbooks, parsed with PyMuPDF)
#
# For Module 1: we hardcode them here so you can run the system
# without any external document store.

DOCUMENTS = [
    {
        "document_id": "runbook-oomkilled-v3",
        "source": "runbooks/oomkilled.md",
        "category": "runbook",
        "k8s_version": "1.29",
        "content": """# OOMKilled Pod Runbook

## What is OOMKilled?

OOMKilled (exit code 137) means the container was forcibly terminated by the Linux
kernel's OOM (Out Of Memory) killer. This happens when a container exceeds its
memory limit as defined in the pod specification. It is NOT an application crash --
it is the operating system killing the process because it consumed too much memory.

Exit code 137 = 128 + 9 (SIGKILL signal number). You will see this in
kubectl describe pod under "Last State: Terminated, Reason: OOMKilled, Exit Code: 137".

## How to Diagnose OOMKilled

Step 1: Confirm the cause.
Run: kubectl describe pod <pod-name> -n <namespace>
Look for the section "Last State" which shows:
  Last State: Terminated
    Reason:   OOMKilled
    Exit Code: 137
    Started:  <timestamp>
    Finished: <timestamp>

Step 2: Check current and peak memory usage.
Run: kubectl top pod <pod-name> -n <namespace>
This shows CURRENT memory usage. Note that the OOM kill happens at PEAK usage,
which may be much higher than current steady-state.

Step 3: Check the memory limit that was enforced.
Run: kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[*].resources}'
This shows both requests and limits. The OOM kill fires when usage reaches the limit.

Step 4: Check application logs from the killed container.
Run: kubectl logs <pod-name> -n <namespace> --previous
The --previous flag (or -p) shows logs from the PREVIOUS container instance.
Without --previous, you see the NEW container's logs (from after restart), not the crashed one.
Look for memory allocation errors, heap dumps, or activity patterns before the kill.

## How to Fix OOMKilled

Option 1: Increase the memory limit (immediate fix).
Edit the deployment:
  kubectl edit deployment <deployment-name> -n <namespace>
Find the resources section and increase limits.memory:
  resources:
    requests:
      memory: "256Mi"
    limits:
      memory: "512Mi"  # increase this value
Start with 1.5x your observed peak usage. Monitor for one week before reducing.

Option 2: Optimize application memory usage (permanent fix).
Profile the application to find memory leaks or inefficient allocations.
Common causes in Java/JVM apps: missing -Xmx flag, heap not configured for container.
Common causes in Python: unbounded caches, large dataframes held in memory.
Common causes in Go: goroutine leaks, unclosed connections accumulating.

Option 3: Set a memory request equal to limit (Guaranteed QoS class).
When requests == limits, Kubernetes places the pod in the Guaranteed QoS class.
This prevents the node from over-committing memory to this pod.
The node scheduler will only place the pod on a node with sufficient free memory.

## Memory Limit Best Practices

Always set both requests AND limits for memory.
Never set limits without requests -- the pod gets Burstable QoS and can be OOMKilled
even when the node has plenty of memory (other pods take priority).

Recommended formula:
  requests.memory = p95 memory usage over the past 7 days
  limits.memory = p99 memory usage * 1.5 safety factor

Use Vertical Pod Autoscaler (VPA) in recommendation mode to get data-driven suggestions:
  kubectl describe vpa <vpa-name> -n <namespace>
VPA shows recommended requests and limits based on actual usage history.

## Related Alerts

Alert: KubePodOOMKilled fires when a pod has been OOMKilled in the last 15 minutes.
Alert: KubeMemoryOvercommit fires when total memory requests exceed node capacity.
Alert: ContainerMemoryUsageHigh fires when usage exceeds 80% of limit (pre-OOM warning).

## Escalation

If OOMKilled recurs more than 3 times in 1 hour despite increasing limits,
escalate to the platform team (platform@company.com) with:
1. Output of: kubectl describe pod <pod-name>
2. Output of: kubectl top pod --all-namespaces | grep <service-name>
3. Application team contact for memory profiling assistance
""",
    },
    {
        "document_id": "runbook-crashloopbackoff-v2",
        "source": "runbooks/crashloopbackoff.md",
        "category": "runbook",
        "k8s_version": "1.29",
        "content": """# CrashLoopBackOff Runbook

## What is CrashLoopBackOff?

CrashLoopBackOff is NOT an error itself -- it is Kubernetes telling you that a container
keeps crashing repeatedly and Kubernetes is backing off (waiting progressively longer)
before each restart attempt.

The backoff schedule is exponential: 10s, 20s, 40s, 80s, 160s, then capped at 5 minutes.
After enough restarts, the pod shows CrashLoopBackOff in its status.

The ACTUAL error is hidden inside the container logs from before it crashed.
CrashLoopBackOff is the symptom. You must find the cause in the logs.

## How to Diagnose CrashLoopBackOff

Step 1: Get the PREVIOUS container's logs (the one that crashed).
Run: kubectl logs <pod-name> -n <namespace> --previous
CRITICAL: Do NOT omit --previous. Without it, you see logs from the current
container (which may have barely started), not the one that crashed.
If the container crashes very quickly, there may be few or no logs.

Step 2: Describe the pod for events and last state.
Run: kubectl describe pod <pod-name> -n <namespace>
Look at:
  "Last State" section: shows exit code (0=clean exit, 1=error, 137=OOMKilled, etc.)
  "Events" section: shows recent Kubernetes events (image pull errors, scheduling issues)

Step 3: Check common exit codes.
Exit code 0:   Container exited cleanly. The process finished and did not restart itself.
               Should your container run forever? Use a Deployment, not a Job.
Exit code 1:   Application error. Check logs for exception stack traces.
Exit code 137: OOMKilled (see OOMKilled runbook).
Exit code 139: Segmentation fault. Native code bug or corrupted binary.
Exit code 143: SIGTERM received but process didn't exit. Graceful shutdown issue.

Step 4: Check if the image exists and can be pulled.
kubectl get events -n <namespace> | grep <pod-name> | grep -i "pull\|image"
"ImagePullBackOff" or "ErrImagePull" in events = image pull failure.
Verify the image tag exists in the registry: docker pull <image:tag>

Step 5: Check for missing ConfigMaps or Secrets.
Run: kubectl get pod <pod-name> -n <namespace> -o yaml | grep -A5 "envFrom\|env:"
If the pod references a ConfigMap or Secret that doesn't exist, it fails immediately.
Events will show: "Error: configmap not found" or "secret not found"

## Common Causes and Fixes

Cause 1: Missing environment variable.
Symptom: Logs show "KeyError: 'DATABASE_URL'" or similar.
Fix: Add the missing env var to the deployment spec or reference the correct Secret.

Cause 2: Database connection refused at startup.
Symptom: Logs show "connection refused" or "dial tcp ... no route to host".
Fix: Add an initContainer with a readiness check, or implement retry logic in the app.
Kubernetes does not guarantee that a pod's dependencies are ready when it starts.

Cause 3: Application exits with code 0 (intentional clean exit).
Symptom: Last State exit code 0, but pod keeps restarting.
Fix: Ensure the container's main process runs indefinitely. Wrap scripts in a loop
or use a process supervisor. A container with exit code 0 triggers a restart if
restartPolicy is Always (the default for Deployments).

Cause 4: Liveness probe failing.
Symptom: Logs look healthy, but pod restarts. Events show "Liveness probe failed".
Fix: Run: kubectl describe pod <pod-name> | grep -A10 "Liveness"
Adjust the initialDelaySeconds to give the application time to start.
Verify the probe endpoint actually works: kubectl exec <pod> -- curl localhost:8080/health

Cause 5: Resource limits too low causing OOM before OOMKilled is reported.
Some containers crash with exit code 1 when they can't allocate memory,
before the OOM killer sends SIGKILL. Check if memory usage was near the limit.

## Escalation Path

If you cannot identify the cause after following all steps:
1. Capture: kubectl logs <pod> --previous > crash-logs.txt
2. Capture: kubectl describe pod <pod> > pod-describe.txt
3. File a ticket with the application team attaching both files.
4. If the service is critical and down: page the on-call engineer via PagerDuty.
""",
    },
    {
        "document_id": "guide-resource-limits-v4",
        "source": "guides/resource-limits.md",
        "category": "guide",
        "k8s_version": "1.29",
        "content": """# Kubernetes Resource Requests and Limits Guide

## Core Concepts

Kubernetes uses two resource directives for CPU and memory:

requests: The amount guaranteed to the container. Used by the scheduler to
find a node with sufficient available capacity. If you set memory request to 256Mi,
Kubernetes will only place the pod on a node that has at least 256Mi free.

limits: The maximum the container is allowed to consume. Exceeding this:
  - For memory: the container is OOMKilled (killed immediately, no warning).
  - For CPU: the container is throttled (slowed down, NOT killed).

## QoS Classes

Kubernetes assigns a Quality of Service class based on your resource configuration.

Guaranteed QoS (best for production):
  requirements.memory == limits.memory AND requests.cpu == limits.cpu
  These pods are the LAST to be evicted when a node runs low on resources.
  They are never throttled unless the CPU limit is reached.

Burstable QoS (default for most workloads):
  At least one resource has requests set, but requests != limits.
  These pods can burst above their requests up to their limits.
  They are evicted before Guaranteed pods during node pressure.

BestEffort QoS (dangerous for production):
  No requests or limits set at all.
  These pods are the FIRST to be evicted during node pressure.
  Never use BestEffort for production workloads.

## Setting Resource Values

Memory recommendations:
  Set requests.memory = observed p95 usage (from kubectl top pod over 7 days)
  Set limits.memory = observed p99 usage * 1.5 (for safety headroom)

CPU recommendations:
  Set requests.cpu = average usage (from metrics-server or Prometheus)
  Set limits.cpu = 2x-4x average usage (allow bursting for latency-sensitive apps)
  Consider NOT setting CPU limits for latency-sensitive services -- CPU throttling
  adds latency even when the node has spare capacity.

Example pod resource configuration:
  resources:
    requests:
      memory: "256Mi"
      cpu: "100m"       # 100 millicores = 0.1 CPU core
    limits:
      memory: "512Mi"
      cpu: "500m"       # 500 millicores = 0.5 CPU core

## Resource Units

Memory units:
  Mi = Mebibytes (1Mi = 1,048,576 bytes) -- PREFERRED in K8s
  Gi = Gibibytes (1Gi = 1,073,741,824 bytes)
  M  = Megabytes (1M = 1,000,000 bytes) -- avoid, confusing
  Note: "256Mi" and "256M" are NOT the same. Always use Mi/Gi.

CPU units:
  m = millicores (1000m = 1 CPU core)
  1 = 1 CPU core
  0.5 = 500m = 0.5 CPU core
  Note: fractional CPU (0.1) is the same as 100m. Both are valid.

## Checking Current Resource Usage

View resource usage for all pods in a namespace:
  kubectl top pods -n <namespace>

View resource usage by node:
  kubectl top nodes

View configured requests and limits for a deployment:
  kubectl get deployment <name> -n <namespace> -o jsonpath='{.spec.template.spec.containers[*].resources}'

View aggregate resource usage vs limits for the namespace:
  kubectl describe resourcequota -n <namespace>

## Namespace Resource Quotas

Platform teams set ResourceQuota objects to prevent any single namespace
from consuming all cluster resources.

View your namespace's quota:
  kubectl get resourcequota -n <namespace>
  kubectl describe resourcequota -n <namespace>

If your deployment fails with "exceeded quota", you have hit the namespace limit.
Request a quota increase from the platform team (platform@company.com).

## LimitRange (Default Limits)

Namespaces may have LimitRange objects that set default requests and limits
for containers that don't specify their own.

View LimitRange:
  kubectl get limitrange -n <namespace>
  kubectl describe limitrange -n <namespace>

If you don't set resources, the LimitRange defaults apply.
Always set explicit resources -- don't rely on defaults.

## Using Vertical Pod Autoscaler (VPA)

VPA automatically recommends or applies resource adjustments based on usage history.
Three modes:
  Off:        VPA calculates recommendations but does not apply them (safe to start with)
  Initial:    VPA sets resources at pod creation but does not update running pods
  Auto:       VPA recreates pods when recommendations change significantly (may cause downtime)

View VPA recommendations:
  kubectl describe vpa <vpa-name> -n <namespace>
Look for the "Recommendation" section showing Lower Bound, Target, and Upper Bound.
Use Target as your requests, Upper Bound * 1.2 as your limits.
""",
    },
    {
        "document_id": "guide-pod-debugging-v2",
        "source": "guides/pod-debugging.md",
        "category": "guide",
        "k8s_version": "1.29",
        "content": """# Pod Debugging Guide

## Universal Debugging Workflow

When any pod is not working correctly, follow this sequence in order.
Do not skip steps -- each one narrows down the problem space.

## Step 1: Check Pod Status

kubectl get pod <pod-name> -n <namespace>

Possible STATUS values and what they mean:
  Pending:              Pod is waiting to be scheduled to a node.
  ContainerCreating:    Pod is scheduled; image is being pulled or volumes mounted.
  Running:              All containers are running. May still be unhealthy.
  Completed:            All containers exited with code 0. Expected for Jobs.
  CrashLoopBackOff:     Container keeps crashing. See CrashLoopBackOff runbook.
  OOMKilled:            Container was killed for exceeding memory limit. See OOMKilled runbook.
  ImagePullBackOff:     Container image cannot be pulled from registry.
  Terminating:          Pod is being deleted. Check for stuck finalizers if it hangs.
  Unknown:              Node lost contact with API server. Check node health.

## Step 2: Describe the Pod

kubectl describe pod <pod-name> -n <namespace>

Key sections to check:
  Conditions:    Ready=True means all containers passed readiness probes.
  Containers:    State (Running/Waiting/Terminated), Restart Count, Last State.
  Events:        Last 60 minutes of events. Start here for Pending pods.
                 Common events: FailedScheduling, Pulling, Pulled, Started, BackOff.

## Step 3: Check Container Logs

For currently running containers:
  kubectl logs <pod-name> -n <namespace>
  kubectl logs <pod-name> -n <namespace> -c <container-name>  # multi-container pods

For the previous (crashed) container:
  kubectl logs <pod-name> -n <namespace> --previous

Follow logs in real time:
  kubectl logs <pod-name> -n <namespace> -f

Get last N lines:
  kubectl logs <pod-name> -n <namespace> --tail=100

## Step 4: Execute Into the Container

If the container is running, open an interactive shell to inspect it:
  kubectl exec -it <pod-name> -n <namespace> -- /bin/bash
  kubectl exec -it <pod-name> -n <namespace> -- /bin/sh  # if bash not available

Inside the container, you can:
  - Check environment variables: env | grep DATABASE
  - Test network connectivity: curl http://service-name:8080/health
  - Check disk space: df -h
  - Check open file handles: ls -la /proc/1/fd | wc -l
  - Check DNS resolution: nslookup kubernetes.default

If the container crashes too fast for exec:
  Create a debug pod with the same image but override the command:
  kubectl debug <pod-name> -n <namespace> -it --copy-to=debug-pod -- /bin/bash

## Step 5: Check Node Health

If pods are Pending or failing to schedule, the problem might be the node.
  kubectl get nodes
  kubectl describe node <node-name>

Look for:
  Conditions: MemoryPressure, DiskPressure, PIDPressure should be False.
  Allocated resources: Check if the node is at or near its capacity.
  Events: Look for disk pressure events, kubelet restarts.

View all events across the cluster sorted by time:
  kubectl get events --all-namespaces --sort-by=.lastTimestamp | tail -30

## Common Quick Diagnoses

Pod Pending for more than 5 minutes:
  kubectl describe pod <name> | grep -A10 Events
  Common causes: insufficient CPU/memory on all nodes, node selector mismatch,
  taint/toleration mismatch, PVC pending, ResourceQuota exceeded.

Pod Running but service not reachable:
  Check readiness probe: kubectl describe pod <name> | grep -A5 Readiness
  Check endpoints: kubectl get endpoints <service-name> -n <namespace>
  An empty endpoints list means no pods are passing the readiness probe.

Service not resolving:
  Check DNS: kubectl exec -it <any-pod> -- nslookup <service-name>
  Check service: kubectl get service <name> -n <namespace>
  Check label selector matches pod labels: kubectl get pod --show-labels

## kubectl Debugging Plugins

Install kubectl-debug for advanced container debugging:
  kubectl debug <pod> -it --image=busybox -n <namespace>
  This attaches a sidecar debug container without restarting the main container.

Install kubectl-neat for cleaner YAML output:
  kubectl get pod <name> -o yaml | kubectl-neat
  Removes managed fields and status noise from YAML output.
""",
    },
    {
        "document_id": "runbook-node-issues-v2",
        "source": "runbooks/node-issues.md",
        "category": "runbook",
        "k8s_version": "1.29",
        "content": """# Node Issues Runbook

## Node NotReady

When a node shows NotReady status, all pods on that node may be evicted or
rescheduled to other nodes after a timeout (default: 5 minutes via pod-eviction-timeout).

Diagnosis:
  kubectl get nodes
  kubectl describe node <node-name>

Check the Conditions section:
  MemoryPressure:  True = node is low on memory, evicting BestEffort pods
  DiskPressure:    True = node disk is nearly full, evicting pods
  PIDPressure:     True = node is running out of process IDs (very rare)
  Ready:           False = kubelet not responding to API server

Check kubelet logs on the node (SSH required):
  journalctl -u kubelet -n 100 --no-pager

Common causes of NotReady:
  1. Node ran out of disk space (common on logging-heavy nodes)
  2. Kubelet crashed and was not restarted automatically
  3. Node network partition (API server can't reach kubelet)
  4. Node hardware failure

## DiskPressure

Node DiskPressure means the node's filesystem is above the eviction threshold (default 85% full).
Kubernetes will evict BestEffort pods and stop scheduling new pods to this node.

Find what is consuming disk space:
  SSH to the node, then:
  df -h
  du -sh /var/log/containers/*    # container logs
  du -sh /var/lib/docker/*        # Docker layers and overlay filesystem
  du -sh /tmp/*

Common causes:
  1. Container logs not rotated (configure log rotation in container runtime)
  2. Large container images accumulated (docker system prune / crictl rmi)
  3. Application writing to / (root filesystem) instead of a mounted volume
  4. Core dumps filling the filesystem

Fix for immediate relief:
  crictl rmi --prune   # remove unused container images (containerd)
  docker system prune  # remove unused images and containers (Docker)

## MemoryPressure

Node MemoryPressure fires when available memory drops below the eviction threshold.
Default threshold: 100Mi remaining.
Kubernetes evicts BestEffort pods first, then Burstable pods.

Diagnose which processes are consuming memory:
  SSH to the node:
  free -m
  cat /proc/meminfo | grep MemAvailable
  ps aux --sort=-%mem | head -20

Force-drain the node to evict all pods safely:
  kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data
  kubectl uncordon <node-name>   # re-enable scheduling after investigation

## Node Cordoning and Draining

Cordon a node (stop new pods from being scheduled, existing pods stay):
  kubectl cordon <node-name>

Drain a node (cordon + evict all evictable pods):
  kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data
  --ignore-daemonsets: DaemonSet pods cannot be evicted (they're on every node by design)
  --delete-emptydir-data: allow draining pods that use emptyDir volumes (data is lost)

Re-enable a cordoned or drained node:
  kubectl uncordon <node-name>

## High Node CPU

High node CPU affects all pods on the node but does NOT cause OOMKilled or evictions.
Instead, it causes CPU throttling for pods at their CPU limits.

Diagnose:
  kubectl top nodes
  kubectl top pods -n <namespace> --sort-by=cpu | head -20

Find which pods are CPU-heavy:
  kubectl top pods --all-namespaces --sort-by=cpu | head -20

If a single pod is consuming excessive CPU:
  kubectl describe pod <pod-name> | grep -A5 "cpu"
  Consider adding CPU limits or investigating a CPU-heavy job/loop in the application.

## Node Replacement

If a node has a hardware issue and must be replaced:
  1. kubectl cordon <node-name>   # prevent new scheduling
  2. kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data
  3. Terminate the instance in your cloud provider console
  4. The node group autoscaler will provision a replacement node (if configured)
  5. Monitor: kubectl get nodes --watch
""",
    },
    {
        "document_id": "guide-rbac-v1",
        "source": "guides/rbac-troubleshooting.md",
        "category": "guide",
        "k8s_version": "1.29",
        "content": """# RBAC Troubleshooting Guide

## Understanding RBAC Errors

RBAC (Role-Based Access Control) errors appear as HTTP 403 Forbidden responses
from the Kubernetes API server. In kubectl, they look like:
  Error from server (Forbidden): pods is forbidden: User "alice" cannot list
  resource "pods" in API group "" in the namespace "production"

The error message tells you exactly what permission is missing:
  User/ServiceAccount: "alice" (the subject)
  Verb:               "list" (the action attempted)
  Resource:           "pods" (the API resource)
  Namespace:          "production" (or "cluster-wide" for ClusterRole)

## Diagnosing RBAC Issues

Step 1: Check what permissions the user or serviceaccount has.
  kubectl auth can-i list pods --as=alice -n production
  Response: "yes" or "no"

Step 2: Test a specific verb and resource.
  kubectl auth can-i create deployments --as=system:serviceaccount:staging:my-sa -n staging

Step 3: Get all permissions for a user.
  kubectl auth can-i --list --as=alice -n production
  This lists all allowed verbs and resources for the specified user in that namespace.

Step 4: Find which RoleBindings apply to this user.
  kubectl get rolebindings,clusterrolebindings --all-namespaces \
    -o custom-columns='NAMESPACE:.metadata.namespace,NAME:.metadata.name,SUBJECTS:.subjects'

## Common RBAC Fixes

Fix 1: Grant namespace-scoped access (Role + RoleBinding).
Create a Role in the target namespace, then bind it to the user:

  apiVersion: rbac.authorization.k8s.io/v1
  kind: Role
  metadata:
    name: pod-reader
    namespace: production
  rules:
  - apiGroups: [""]
    resources: ["pods", "pods/log"]
    verbs: ["get", "list", "watch"]

  apiVersion: rbac.authorization.k8s.io/v1
  kind: RoleBinding
  metadata:
    name: pod-reader-alice
    namespace: production
  subjects:
  - kind: User
    name: alice
    apiGroup: rbac.authorization.k8s.io
  roleRef:
    kind: Role
    name: pod-reader
    apiGroup: rbac.authorization.k8s.io

Fix 2: Grant cluster-wide access (ClusterRole + ClusterRoleBinding).
Use this for permissions that span all namespaces or for non-namespaced resources.
ClusterRoleBindings cannot be restricted to a namespace.

Fix 3: ServiceAccount permissions (for pods accessing the API).
If a pod needs to call the Kubernetes API (e.g., an operator):
  kubectl create serviceaccount my-sa -n my-namespace
  kubectl create rolebinding my-sa-binding \
    --role=pod-reader --serviceaccount=my-namespace:my-sa -n my-namespace

## Verifying RBAC Changes

After creating/updating RBAC resources:
  kubectl auth can-i <verb> <resource> --as=<user> -n <namespace>

RBAC changes take effect immediately -- no restart needed.
The API server re-evaluates RBAC on every request.

## ServiceAccount Token in Pods

Pods automatically mount a ServiceAccount token at:
  /var/run/secrets/kubernetes.io/serviceaccount/token

To disable automatic token mounting (security best practice for pods that don't need API access):
  spec:
    automountServiceAccountToken: false

To use a custom ServiceAccount:
  spec:
    serviceAccountName: my-sa
""",
    },
    {
        "document_id": "guide-networking-v2",
        "source": "guides/networking-issues.md",
        "category": "guide",
        "k8s_version": "1.29",
        "content": """# Kubernetes Networking Troubleshooting Guide

## Pod-to-Pod Connectivity

All pods in a Kubernetes cluster can communicate with all other pods by default
(unless NetworkPolicy restricts this). Each pod gets a unique cluster IP.

Test pod-to-pod connectivity:
  kubectl exec -it <pod-a> -n <namespace> -- curl http://<pod-b-ip>:8080
  kubectl exec -it <pod-a> -n <namespace> -- ping <pod-b-ip>

Get pod IP addresses:
  kubectl get pods -n <namespace> -o wide   # shows pod IPs

## Service DNS Resolution

Kubernetes Services are accessible by DNS name within the cluster.
Full DNS format: <service-name>.<namespace>.svc.cluster.local
Short format:    <service-name> (within the same namespace)

Test DNS resolution:
  kubectl exec -it <pod> -n <namespace> -- nslookup my-service
  kubectl exec -it <pod> -n <namespace> -- nslookup my-service.other-namespace.svc.cluster.local

DNS not resolving? Check CoreDNS:
  kubectl get pods -n kube-system | grep coredns
  kubectl logs -n kube-system <coredns-pod>

## Service Not Reachable

If you can reach the pod IP directly but not the service IP:

Step 1: Check the service exists and has the right port.
  kubectl get service <service-name> -n <namespace>
  kubectl describe service <service-name> -n <namespace>

Step 2: Check the service has endpoints (pods that pass readiness probes).
  kubectl get endpoints <service-name> -n <namespace>
  An empty endpoints list (no IPs) means NO pods are passing readiness probes.
  Fix: Check pod readiness probes and pod status.

Step 3: Verify the service selector matches pod labels.
  kubectl get service <service-name> -o yaml | grep -A5 selector
  kubectl get pods -n <namespace> --show-labels | grep <expected-label>
  The service selector must match pod labels exactly (key AND value).

## NetworkPolicy

NetworkPolicy objects restrict which pods can communicate.
If connectivity was working and broke after a NetworkPolicy was applied:

List NetworkPolicies in namespace:
  kubectl get networkpolicies -n <namespace>
  kubectl describe networkpolicy <name> -n <namespace>

Check if a NetworkPolicy is blocking traffic:
  If a namespace has any NetworkPolicy, ALL traffic not explicitly allowed is denied.
  A pod with no matching NetworkPolicy is fully open (default allow all).

Test with network policy disabled temporarily (for debugging only):
  kubectl delete networkpolicy <name> -n <namespace>  # CAREFUL in production!

## Common Networking Issues

Connection refused (port not open):
  The service exists but nothing is listening on the target port.
  kubectl exec -it <pod> -- ss -tlnp   # check listening ports inside the pod

Connection timed out (firewall or no route):
  Network path is blocked. Check cloud security groups, VPC routes, NetworkPolicy.
  kubectl exec -it <pod> -- traceroute <destination>

DNS SERVFAIL:
  CoreDNS is running but returning errors. Check CoreDNS logs.
  Common cause: upstream DNS (for external names) is unreachable.
  kubectl logs -n kube-system -l k8s-app=kube-dns | tail -50

## Ingress Troubleshooting

Check if Ingress controller pods are running:
  kubectl get pods -n ingress-nginx   # or your ingress controller namespace

Check Ingress resource:
  kubectl describe ingress <name> -n <namespace>
  Look for "Events" section -- ingress controller logs errors here.

Common Ingress issues:
  404: The path rule doesn't match, or the backend service doesn't exist.
  502/503: The backend service exists but pods are not ready.
  SSL errors: Certificate Secret is wrong name or in wrong namespace.

View ingress controller logs:
  kubectl logs -n ingress-nginx <ingress-controller-pod> | tail -100
""",
    },
    {
        "document_id": "guide-deployments-v3",
        "source": "guides/deployment-rollback.md",
        "category": "guide",
        "k8s_version": "1.29",
        "content": """# Deployment Management and Rollback Guide

## Deployment Rolling Update

By default, Kubernetes Deployments use rolling updates:
  1. New pods are created with the new image/config
  2. Once new pods are Ready, old pods are terminated
  3. Default: 25% max unavailable, 25% max surge

Monitor a rollout in progress:
  kubectl rollout status deployment/<name> -n <namespace>
  kubectl get pods -n <namespace> -w   # watch pods being replaced

## Rollback a Failed Deployment

If a deployment causes issues, roll back immediately:

Step 1: Check rollout history.
  kubectl rollout history deployment/<name> -n <namespace>
  This shows revision numbers and change causes.

Step 2: Roll back to the previous revision.
  kubectl rollout undo deployment/<name> -n <namespace>
  This instantly triggers a new rollout using the previous ReplicaSet.

Step 3: Roll back to a specific revision.
  kubectl rollout undo deployment/<name> --to-revision=3 -n <namespace>

Step 4: Verify rollback is complete.
  kubectl rollout status deployment/<name> -n <namespace>
  kubectl get pods -n <namespace>

## Pause and Resume Rollouts

Pause a rollout (to apply multiple changes without triggering intermediate rollouts):
  kubectl rollout pause deployment/<name> -n <namespace>
  # make your changes
  kubectl rollout resume deployment/<name> -n <namespace>

## Canary Deployments (Manual)

For production changes, use a canary deployment:
1. Scale down production deployment to 90% of desired replicas.
2. Create a new deployment with the new image at 10% of desired replicas.
3. Monitor error rates and latency for 30 minutes.
4. If healthy: scale up new, scale down old.
5. If unhealthy: scale down new, scale up old. Zero downtime.

## Checking Deployment Health

View deployment status:
  kubectl get deployment <name> -n <namespace>
  READY column shows: <ready-replicas>/<desired-replicas>
  AVAILABLE column shows replicas passing readiness probes.

View pods controlled by a deployment:
  kubectl get pods -n <namespace> -l app=<app-label>

View ReplicaSets (deployment revision history):
  kubectl get replicasets -n <namespace> -l app=<app-label>
  Each ReplicaSet corresponds to one deployment revision.

## Deployment Strategies

RollingUpdate (default): Zero-downtime, gradual replacement.
  Adjust with: maxSurge (extra pods allowed) and maxUnavailable (pods allowed to be down).
  Good for: Most services.

Recreate: All old pods deleted before new pods are created. Causes downtime.
  Use when: New version is incompatible with old version and cannot run side by side.
  Example: Database schema migrations that break the old application version.

Blue-Green (manual): Two full deployments active, switch traffic at once.
  Use when: You need instant rollback capability and have capacity to run 2x replicas.

## Zero-Downtime Deployment Checklist

Before deploying:
  1. Readiness probe is configured and tested.
     Without readiness probes, Kubernetes sends traffic to pods before they're ready.
  2. Liveness probe is configured with appropriate initialDelaySeconds.
     Too short = liveness probe kills healthy pods still warming up.
  3. Graceful shutdown is implemented (handle SIGTERM signal).
     Kubernetes sends SIGTERM before terminating a pod. Application must finish
     in-flight requests within terminationGracePeriodSeconds (default 30s).
  4. PodDisruptionBudget (PDB) is configured.
     kubectl get pdb -n <namespace>
     A PDB ensures at least N replicas are always available during disruptions.
  5. Resource requests and limits are set appropriately.
     Insufficient resources = pod evicted during high-load rollouts.

## Image Tag Best Practices

Never use `latest` tag in production deployments.
Using `latest` makes rollbacks impossible -- you can't tell which image was deployed.
Always use immutable tags: semantic versions (v1.2.3) or git commit SHAs.

Check what image is currently running:
  kubectl get deployment <name> -n <namespace> -o jsonpath='{.spec.template.spec.containers[*].image}'
""",
    },
    {
        "document_id": "guide-monitoring-v2",
        "source": "guides/monitoring-with-kubectl.md",
        "category": "guide",
        "k8s_version": "1.29",
        "content": """# Kubernetes Monitoring and Observability Guide

## Real-Time Resource Usage

View current CPU and memory usage for all pods in a namespace:
  kubectl top pods -n <namespace>
  kubectl top pods -n <namespace> --sort-by=memory   # sort by memory usage
  kubectl top pods -n <namespace> --sort-by=cpu       # sort by CPU usage

View node resource usage:
  kubectl top nodes

Note: kubectl top requires the metrics-server to be installed in the cluster.
If you get "error: Metrics API not available", run:
  kubectl get pods -n kube-system | grep metrics-server

## Watching Resources in Real Time

Watch pod status change in real time:
  kubectl get pods -n <namespace> -w

Watch events stream in real time:
  kubectl get events -n <namespace> -w

Watch with a wider output (shows more columns):
  kubectl get pods -n <namespace> -o wide -w

## Cluster-Wide Event Search

View all events sorted by time (newest last):
  kubectl get events --all-namespaces --sort-by=.lastTimestamp

View only Warning events:
  kubectl get events --all-namespaces --field-selector type=Warning

View events for a specific pod:
  kubectl get events -n <namespace> --field-selector involvedObject.name=<pod-name>

## Log Aggregation

For production workloads, logs must be shipped to a centralized log store.
Our stack uses Fluentd (DaemonSet on every node) -> Elasticsearch -> Kibana.

Access Kibana:  https://kibana.internal.company.com
Default index:  kubernetes-*
Filter by:      kubernetes.namespace_name, kubernetes.pod_name, kubernetes.container_name

For quick local debugging (without Kibana):
  kubectl logs <pod> -n <namespace> --tail=200 | grep -i "error\|exception\|fatal"

Stream logs from all pods matching a label:
  kubectl logs -n <namespace> -l app=payment-service -f --prefix
  The --prefix flag shows which pod each log line came from.

## Prometheus Metrics

Our Prometheus instance is at: https://prometheus.internal.company.com
Our Grafana dashboards are at:  https://grafana.internal.company.com

Useful PromQL queries for K8s operations:

Pod restarts in last hour:
  increase(kube_pod_container_status_restarts_total{namespace="production"}[1h]) > 0

Memory usage vs limit:
  container_memory_working_set_bytes / container_spec_memory_limit_bytes

CPU throttling rate:
  rate(container_cpu_throttled_seconds_total[5m]) / rate(container_cpu_usage_seconds_total[5m])

Node disk usage:
  (node_filesystem_size_bytes - node_filesystem_avail_bytes) / node_filesystem_size_bytes

## Alerts

All production alerts are routed through PagerDuty.
Critical alerts page the on-call engineer immediately.
Warning alerts are sent to Slack #platform-alerts.

Key production alerts and their runbooks:
  KubePodOOMKilled          -> See oomkilled.md runbook
  KubePodCrashLooping       -> See crashloopbackoff.md runbook
  KubeNodeNotReady          -> See node-issues.md runbook
  KubeDeploymentReplicasMismatch -> Check if pods are stuck in Pending/Error
  KubePersistentVolumeErrors -> Storage issue, page on-call immediately

View current firing alerts:
  https://alertmanager.internal.company.com
  Or: kubectl port-forward -n monitoring svc/alertmanager 9093:9093

## Health Check Endpoints

All production services expose health endpoints:
  /health         Basic health check (returns 200 if running)
  /ready          Readiness check (returns 200 only if ready to serve traffic)
  /metrics        Prometheus metrics endpoint

Test health checks from inside the cluster:
  kubectl exec -it <any-pod> -n <namespace> -- curl http://<service-name>:8080/health
  kubectl exec -it <any-pod> -n <namespace> -- curl http://<service-name>:8080/ready

If /health returns 200 but /ready returns 503: the pod is running but not yet ready.
This is normal during startup. Abnormal if it persists after 60+ seconds.
""",
    },
    {
        "document_id": "guide-ingress-v2",
        "source": "guides/ingress.md",
        "category": "guide",
        "k8s_version": "1.29",
        "content": """# Ingress Configuration Guide

## What is Ingress?

Ingress is a Kubernetes API object that manages external HTTP/HTTPS access to services
inside the cluster. It provides:
  - Host-based routing: api.company.com -> api-service, web.company.com -> web-service
  - Path-based routing: /api/* -> api-service, /static/* -> static-service
  - TLS termination: HTTPS at the edge, HTTP inside the cluster
  - Load balancing across pods behind each service

Our cluster uses nginx-ingress-controller as the Ingress implementation.
All Ingress resources use: kubernetes.io/ingress.class: "nginx"

## Basic Ingress Example

  apiVersion: networking.k8s.io/v1
  kind: Ingress
  metadata:
    name: my-ingress
    namespace: production
    annotations:
      kubernetes.io/ingress.class: "nginx"
      nginx.ingress.kubernetes.io/rewrite-target: /
  spec:
    tls:
    - hosts:
      - api.company.com
      secretName: api-company-com-tls   # Secret with TLS certificate
    rules:
    - host: api.company.com
      http:
        paths:
        - path: /
          pathType: Prefix
          backend:
            service:
              name: api-service
              port:
                number: 8080

## TLS Certificate Management

We use cert-manager to automatically provision and renew TLS certificates via Let's Encrypt.

Check certificate status:
  kubectl get certificate -n <namespace>
  kubectl describe certificate <name> -n <namespace>

Certificate READY=True means it's valid and provisioned in the Secret.
Certificate READY=False: check kubectl describe for the reason.
Common reasons: DNS challenge failed, rate limit hit, domain not accessible.

Request a new certificate manually:
  kubectl delete certificate <name> -n <namespace>
  cert-manager will re-create it automatically if a CertificateRequest exists.

## Ingress Annotations (nginx-ingress)

Increase request timeout (default: 60s):
  nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
  nginx.ingress.kubernetes.io/proxy-send-timeout: "300"

Enable CORS:
  nginx.ingress.kubernetes.io/enable-cors: "true"
  nginx.ingress.kubernetes.io/cors-allow-origin: "https://app.company.com"

Rate limiting:
  nginx.ingress.kubernetes.io/limit-rps: "100"          # requests per second
  nginx.ingress.kubernetes.io/limit-connections: "50"   # concurrent connections

Upload size limit (default: 1m):
  nginx.ingress.kubernetes.io/proxy-body-size: "50m"    # for file uploads

## Debugging Ingress

Check Ingress resource is configured correctly:
  kubectl describe ingress <name> -n <namespace>
  Look at Address field: should show the Load Balancer IP.
  Look at Rules section: verify host/path/service mapping is correct.
  Look at Events: ingress controller logs errors here.

Check ingress controller logs:
  kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx | tail -100
  Look for upstream connection errors, SSL handshake failures, 404s.

Test connectivity through ingress:
  curl -H "Host: api.company.com" http://<ingress-lb-ip>/health
  curl https://api.company.com/health

502 Bad Gateway: Backend pods are not accepting connections.
  Check: kubectl get endpoints <backend-service> -n <namespace>
  If endpoints are empty: pods are not passing readiness probe.

503 Service Unavailable: No healthy backend pods.
  Check pod health: kubectl get pods -n <namespace> -l app=<service>
  Check HPA: kubectl get hpa -n <namespace>

SSL certificate error:
  openssl s_client -connect api.company.com:443 -servername api.company.com 2>&1 | grep "subject\|issuer"
  Check the TLS Secret: kubectl get secret <secret-name> -n <namespace>
""",
    },
]


def get_all_documents() -> list[dict]:
    """Return the full knowledge base."""
    return DOCUMENTS


def get_document_by_id(document_id: str) -> dict | None:
    """Look up a document by its ID."""
    for doc in DOCUMENTS:
        if doc["document_id"] == document_id:
            return doc
    return None


if __name__ == "__main__":
    print(f"Knowledge base: {len(DOCUMENTS)} documents")
    for doc in DOCUMENTS:
        word_count = len(doc["content"].split())
        print(f"  {doc['document_id']}: {word_count} words, category={doc['category']}")
