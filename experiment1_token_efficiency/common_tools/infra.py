INFRA_TOOLS = [
    {
        "name": "kubernetes_status",
        "description": "Check the status of Kubernetes pods, services, and deployments in a namespace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Kubernetes namespace to check (e.g. 'production', 'staging')",
                },
                "resource_type": {
                    "type": "string",
                    "description": "Type of resource (pods, services, deployments, all)",
                    "enum": ["pods", "services", "deployments", "all"],
                },
            },
            "required": ["namespace"],
        },
        "mock_response": {
            "namespace": "production",
            "pods": [
                {"name": "payment-api-7d8f9c", "status": "CrashLoopBackOff", "restarts": 5, "age": "2h"},
                {"name": "payment-api-7d8f9b", "status": "Running", "restarts": 0, "age": "2h"},
                {"name": "user-service-3a4b2c", "status": "Running", "restarts": 0, "age": "12h"},
                {"name": "redis-cache-6e7f8a", "status": "Running", "restarts": 1, "age": "48h"},
            ],
            "deployments": [
                {"name": "payment-api", "available": 1, "desired": 2, "status": "degraded"},
                {"name": "user-service", "available": 3, "desired": 3, "status": "healthy"},
            ],
        },
    },
    {
        "name": "deployment_logs",
        "description": "Retrieve recent deployment activity and release history for a service.",
        "input_schema": {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": "Service name to check deployment history",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of recent deployments to return",
                },
            },
            "required": ["service"],
        },
        "mock_response": {
            "service": "payment-api",
            "deployments": [
                {"version": "v2.4.1", "timestamp": "2025-05-21T09:00:00Z", "status": "failed", "reason": "health check timeout"},
                {"version": "v2.4.0", "timestamp": "2025-05-20T14:00:00Z", "status": "success", "reason": ""},
                {"version": "v2.3.2", "timestamp": "2025-05-19T11:00:00Z", "status": "success", "reason": ""},
                {"version": "v2.3.1", "timestamp": "2025-05-18T09:00:00Z", "status": "rollback", "reason": "memory leak detected"},
            ],
        },
    },
    {
        "name": "cpu_metrics",
        "description": "Get CPU usage metrics for services or pods over a time window.",
        "input_schema": {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": "Service name to get CPU metrics for",
                },
                "time_window": {
                    "type": "string",
                    "description": "Time window (e.g. '5m', '1h', '24h')",
                },
            },
            "required": ["service"],
        },
        "mock_response": {
            "service": "payment-api",
            "time_window": "1h",
            "metrics": [
                {"timestamp": "2025-05-21T10:20:00Z", "cpu_percent": 92},
                {"timestamp": "2025-05-21T10:15:00Z", "cpu_percent": 88},
                {"timestamp": "2025-05-21T10:10:00Z", "cpu_percent": 45},
                {"timestamp": "2025-05-21T10:05:00Z", "cpu_percent": 32},
                {"timestamp": "2025-05-21T10:00:00Z", "cpu_percent": 28},
            ],
            "average": 57.0,
            "peak": 92,
        },
    },
]
