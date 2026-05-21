PM_TOOLS = [
    {
        "name": "create_ticket",
        "description": "Create a new ticket or issue in the project management system with a title, description, and priority.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Ticket title",
                },
                "description": {
                    "type": "string",
                    "description": "Detailed description of the issue or task",
                },
                "priority": {
                    "type": "string",
                    "description": "Priority level (low, medium, high, critical)",
                    "enum": ["low", "medium", "high", "critical"],
                },
                "assignee": {
                    "type": "string",
                    "description": "Optional team member to assign to",
                },
            },
            "required": ["title", "description", "priority"],
        },
        "mock_response": {
            "ticket_id": "INC-2025-05-21-0042",
            "status": "created",
            "url": "https://pm.internal/tickets/INC-2025-05-21-0042",
            "created_at": "2025-05-21T10:30:00Z",
        },
    },
    {
        "name": "send_notification",
        "description": "Send a notification to a channel or user. Supports Slack, email, and PagerDuty targets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Notification target (e.g. '#incidents', '@oncall', 'team@email.com')",
                },
                "message": {
                    "type": "string",
                    "description": "Notification message content",
                },
                "urgency": {
                    "type": "string",
                    "description": "Urgency level (info, warning, critical)",
                    "enum": ["info", "warning", "critical"],
                },
            },
            "required": ["channel", "message", "urgency"],
        },
        "mock_response": {
            "status": "sent",
            "channel": "#incidents",
            "delivered_at": "2025-05-21T10:31:00Z",
            "recipients": ["@oncall-team"],
        },
    },
]
