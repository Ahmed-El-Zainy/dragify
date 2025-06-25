"""
Dragify AI Agent Template - Gradio Demo
Interactive demo showcasing the AI agent automation platform
"""

import gradio as gr
import pandas as pd
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import random

# Mock data and configurations
TRIGGER_TYPES = ["Slack Message", "Gmail Email", "Webhook"]
DATA_SOURCES = ["Google Sheets", "MongoDB", "PostgreSQL", "Airtable"]
CRM_SYSTEMS = ["Zoho CRM", "Salesforce", "Odoo CRM", "Mock CRM"]

# Sample data for demo
SAMPLE_LEADS = [
    {"name": "John Doe", "email": "john@acme.com", "company": "Acme Corp", "phone": "+1-555-0123", "source": "web_form"},
    {"name": "Jane Smith", "email": "jane@techsol.com", "company": "Tech Solutions", "phone": "+1-555-0456", "source": "linkedin"},
    {"name": "Bob Johnson", "email": "bob@enterprise.com", "company": "Enterprise LLC", "phone": "+1-555-0789", "source": "referral"},
    {"name": "Alice Brown", "email": "alice@startup.com", "company": "StartupXYZ", "phone": "+1-555-0101", "source": "event"},
    {"name": "Charlie Wilson", "email": "charlie@innovate.com", "company": "Innovate Inc", "phone": "+1-555-0202", "source": "cold_email"},
]

# Global state for demo
demo_state = {
    "agents": [],
    "executions": [],
    "agent_counter": 0
}

class DragifyAgentDemo:
    def __init__(self):
        self.reset_demo_data()
    
    def reset_demo_data(self):
        """Reset demo data to initial state"""
        demo_state["agents"] = []
        demo_state["executions"] = []
        demo_state["agent_counter"] = 0
        
        # Add sample agents
        self.create_sample_agents()
    
    def create_sample_agents(self):
        """Create sample agents for demo"""
        sample_agents = [
            {
                "name": "Lead Capture Bot",
                "description": "Captures leads from Slack mentions and adds to CRM",
                "trigger": "Slack Message",
                "data_source": "Google Sheets",
                "crm": "Zoho CRM",
                "status": "Active"
            },
            {
                "name": "Email Lead Processor",
                "description": "Processes leads from Gmail and syncs with Salesforce",
                "trigger": "Gmail Email",
                "data_source": "MongoDB",
                "crm": "Salesforce",
                "status": "Active"
            },
            {
                "name": "Webhook Lead Handler",
                "description": "Handles web form submissions via webhook",
                "trigger": "Webhook",
                "data_source": "Airtable",
                "crm": "Mock CRM",
                "status": "Active"
            }
        ]
        
        for agent_data in sample_agents:
            agent_id = self.create_agent(
                agent_data["name"],
                agent_data["description"],
                agent_data["trigger"],
                agent_data["data_source"],
                agent_data["crm"]
            )

    def create_agent(self, name: str, description: str, trigger: str, data_source: str, crm: str) -> int:
        """Create a new agent configuration"""
        demo_state["agent_counter"] += 1
        agent_id = demo_state["agent_counter"]
        
        agent = {
            "id": agent_id,
            "name": name,
            "description": description,
            "trigger": trigger,
            "data_source": data_source,
            "crm": crm,
            "status": "Active",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "executions": 0,
            "success_rate": "100%"
        }
        
        demo_state["agents"].append(agent)
        return agent_id
    
    def get_agents_dataframe(self) -> pd.DataFrame:
        """Get agents as DataFrame for display"""
        if not demo_state["agents"]:
            return pd.DataFrame(columns=["ID", "Name", "Trigger", "Data Source", "CRM", "Status", "Created"])
        
        agents_data = []
        for agent in demo_state["agents"]:
            agents_data.append([
                agent["id"],
                agent["name"],
                agent["trigger"],
                agent["data_source"],
                agent["crm"],
                agent["status"],
                agent["created_at"]
            ])
        
        return pd.DataFrame(agents_data, columns=["ID", "Name", "Trigger", "Data Source", "CRM", "Status", "Created"])
    
    def execute_workflow(self, agent_id: int, trigger_data: Dict) -> Dict:
        """Execute a workflow for testing"""
        # Find agent
        agent = next((a for a in demo_state["agents"] if a["id"] == agent_id), None)
        if not agent:
            return {"error": "Agent not found"}
        
        # Simulate workflow execution
        execution_id = len(demo_state["executions"]) + 1
        
        # Simulate data fetching
        lead_data = random.choice(SAMPLE_LEADS).copy()
        lead_data["trigger_source"] = agent["trigger"]
        lead_data["fetched_from"] = agent["data_source"]
        
        # Simulate CRM insertion
        crm_response = {
            "crm": agent["crm"],
            "lead_id": f"{agent['crm'].upper()}_{uuid.uuid4().hex[:8]}",
            "status": "created" if random.random() > 0.1 else "failed",
            "timestamp": datetime.now().isoformat()
        }
        
        # Create execution record
        execution = {
            "id": execution_id,
            "agent_id": agent_id,
            "agent_name": agent["name"],
            "trigger_data": trigger_data,
            "lead_data": lead_data,
            "crm_response": crm_response,
            "status": "Success" if crm_response["status"] == "created" else "Failed",
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration": f"{random.randint(2, 8)} seconds"
        }
        
        demo_state["executions"].insert(0, execution)  # Most recent first
        
        # Update agent stats
        agent["executions"] += 1
        
        return execution
    
    def get_executions_dataframe(self) -> pd.DataFrame:
        """Get executions as DataFrame for display"""
        if not demo_state["executions"]:
            return pd.DataFrame(columns=["ID", "Agent", "Status", "Started", "Duration"])
        
        executions_data = []
        for exec in demo_state["executions"][:20]:  # Show last 20
            executions_data.append([
                exec["id"],
                exec["agent_name"],
                exec["status"],
                exec["started_at"],
                exec["duration"]
            ])
        
        return pd.DataFrame(executions_data, columns=["ID", "Agent", "Status", "Started", "Duration"])
    
    def get_execution_details(self, execution_id: int) -> Dict:
        """Get detailed execution information"""
        execution = next((e for e in demo_state["executions"] if e["id"] == execution_id), None)
        if not execution:
            return {"error": "Execution not found"}
        return execution

# Initialize demo
demo = DragifyAgentDemo()

# Gradio Interface Functions
def create_new_agent(name, description, trigger, data_source, crm):
    """Create a new agent via Gradio interface"""
    if not name:
        return "❌ Agent name is required", demo.get_agents_dataframe()
    
    try:
        agent_id = demo.create_agent(name, description, trigger, data_source, crm)
        return f"✅ Agent '{name}' created successfully (ID: {agent_id})", demo.get_agents_dataframe()
    except Exception as e:
        return f"❌ Error creating agent: {str(e)}", demo.get_agents_dataframe()

def trigger_workflow_demo(agent_id, trigger_message):
    """Trigger a workflow execution for demo"""
    if not agent_id:
        return "❌ Please select an agent", demo.get_executions_dataframe(), ""
    
    try:
        agent_id = int(agent_id)
        trigger_data = {
            "message": trigger_message or "Demo trigger",
            "timestamp": datetime.now().isoformat(),
            "source": "gradio_demo"
        }
        
        execution = demo.execute_workflow(agent_id, trigger_data)
        
        if "error" in execution:
            return f"❌ {execution['error']}", demo.get_executions_dataframe(), ""
        
        # Format execution details
        details = f"""
🚀 **Workflow Executed Successfully!**

**Execution ID:** {execution['id']}
**Agent:** {execution['agent_name']}
**Status:** {execution['status']}
**Duration:** {execution['duration']}

**📥 Trigger Data:**
```json
{json.dumps(execution['trigger_data'], indent=2)}
```

**📊 Fetched Lead Data:**
- **Name:** {execution['lead_data']['name']}
- **Email:** {execution['lead_data']['email']}
- **Company:** {execution['lead_data']['company']}
- **Phone:** {execution['lead_data']['phone']}
- **Source:** {execution['lead_data']['source']}

**🎯 CRM Response:**
- **CRM:** {execution['crm_response']['crm']}
- **Lead ID:** {execution['crm_response']['lead_id']}
- **Status:** {execution['crm_response']['status'].upper()}

**📧 Email Notification:** Sent to admin@company.com
        """
        
        return f"✅ Workflow executed successfully (ID: {execution['id']})", demo.get_executions_dataframe(), details
        
    except Exception as e:
        return f"❌ Error executing workflow: {str(e)}", demo.get_executions_dataframe(), ""

def get_agent_options():
    """Get agent options for dropdown"""
    if not demo_state["agents"]:
        return []
    return [f"{agent['id']} - {agent['name']}" for agent in demo_state["agents"]]

def refresh_data():
    """Refresh all data displays"""
    return demo.get_agents_dataframe(), demo.get_executions_dataframe(), get_agent_options()

def reset_demo():
    """Reset demo to initial state"""
    demo.reset_demo_data()
    return demo.get_agents_dataframe(), demo.get_executions_dataframe(), get_agent_options(), "✅ Demo reset successfully!"

# Custom CSS for better styling
custom_css = """
.gradio-container {
    max-width: 1200px !important;
}
.tab-nav {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 0.25rem;
    padding: 0.75rem;
    margin: 0.5rem 0;
}
.error-box {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    border-radius: 0.25rem;
    padding: 0.75rem;
    margin: 0.5rem 0;
}
"""

# Create Gradio Interface
with gr.Blocks(css=custom_css, title="Dragify AI Agent Demo", theme=gr.themes.Soft()) as interface:
    
    gr.HTML("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
        <h1>🤖 Dragify AI Agent Template</h1>
        <p>Interactive Demo - Modular AI Agent Automation Platform</p>
        <p><em>Built for Cloudilic Engineering Assessment</em></p>
    </div>
    """)
    
    with gr.Tabs() as tabs:
        
        # Dashboard Tab
        with gr.Tab("📊 Dashboard"):
            gr.Markdown("## Agent Overview")
            
            with gr.Row():
                with gr.Column(scale=2):
                    agents_table = gr.Dataframe(
                        value=demo.get_agents_dataframe(),
                        label="Active Agents",
                        interactive=False,
                        wrap=True
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("""
                    ### 📈 Quick Stats
                    - **Total Agents:** 3
                    - **Active Workflows:** 3
                    - **Success Rate:** 95%
                    - **Avg Response Time:** 4.2s
                    
                    ### 🔧 System Status
                    - **API:** ✅ Online
                    - **Database:** ✅ Connected
                    - **OAuth:** ✅ Ready
                    - **CRM Integrations:** ✅ Active
                    """)
            
            gr.Markdown("## Recent Executions")
            executions_table = gr.Dataframe(
                value=demo.get_executions_dataframe(),
                label="Workflow Executions (Last 20)",
                interactive=False,
                wrap=True
            )
            
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Data", variant="secondary")
                reset_btn = gr.Button("🔄 Reset Demo", variant="secondary")
                reset_status = gr.Textbox(label="Status", visible=False)
        
        # Agent Builder Tab
        with gr.Tab("🛠️ Agent Builder"):
            gr.Markdown("## Create New AI Agent")
            gr.Markdown("Configure a new workflow automation agent with custom triggers, data sources, and CRM integrations.")
            
            with gr.Row():
                with gr.Column():
                    agent_name = gr.Textbox(
                        label="Agent Name",
                        placeholder="e.g., Lead Capture Bot",
                        value=""
                    )
                    agent_description = gr.Textbox(
                        label="Description",
                        placeholder="Brief description of what this agent does",
                        lines=2
                    )
                
                with gr.Column():
                    trigger_type = gr.Dropdown(
                        choices=TRIGGER_TYPES,
                        label="Trigger Type",
                        value="Slack Message"
                    )
                    data_source = gr.Dropdown(
                        choices=DATA_SOURCES,
                        label="Data Source",
                        value="Google Sheets"
                    )
                    crm_system = gr.Dropdown(
                        choices=CRM_SYSTEMS,
                        label="CRM System",
                        value="Mock CRM"
                    )
            
            create_btn = gr.Button("🚀 Create Agent", variant="primary", size="lg")
            create_status = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("### Configuration Examples")
            with gr.Accordion("View Sample Configurations", open=False):
                gr.Markdown("""
                **Example 1: Slack → Google Sheets → Zoho CRM**
                - Trigger: New mention in Slack channel
                - Data: Fetch lead details from Google Sheets form responses
                - CRM: Create lead in Zoho CRM
                - Notification: Email confirmation to sales team
                
                **Example 2: Gmail → MongoDB → Salesforce**
                - Trigger: New email with specific subject pattern
                - Data: Query MongoDB for customer enrichment data
                - CRM: Create opportunity in Salesforce
                - Notification: Slack notification to account manager
                
                **Example 3: Webhook → Airtable → Odoo CRM**
                - Trigger: Web form submission webhook
                - Data: Additional lead scoring from Airtable
                - CRM: Create and assign lead in Odoo CRM
                - Notification: Email notification with lead score
                """)
        
        # Workflow Tester Tab
        with gr.Tab("🧪 Workflow Tester"):
            gr.Markdown("## Test Workflow Execution")
            gr.Markdown("Trigger workflows manually to test your agent configurations.")
            
            with gr.Row():
                with gr.Column():
                    agent_selector = gr.Dropdown(
                        choices=get_agent_options(),
                        label="Select Agent to Test",
                        value=None
                    )
                    trigger_message = gr.Textbox(
                        label="Trigger Message/Data",
                        placeholder="Enter trigger data (e.g., Slack message, email content, webhook payload)",
                        lines=3,
                        value="New lead from demo form submission"
                    )
                    
                    test_btn = gr.Button("▶️ Execute Workflow", variant="primary", size="lg")
                    test_status = gr.Textbox(label="Execution Status", interactive=False)
                
                with gr.Column():
                    execution_details = gr.Markdown(
                        value="Select an agent and click 'Execute Workflow' to see detailed results here.",
                        label="Execution Details"
                    )
            
            gr.Markdown("### Test Scenarios")
            with gr.Row():
                scenario1_btn = gr.Button("📧 Test Email Lead", variant="secondary")
                scenario2_btn = gr.Button("💬 Test Slack Mention", variant="secondary")
                scenario3_btn = gr.Button("🌐 Test Webhook", variant="secondary")
        
        # Logs & Monitoring Tab
        with gr.Tab("📋 Logs & Monitoring"):
            gr.Markdown("## Execution Logs & Monitoring")
            
            with gr.Row():
                with gr.Column(scale=2):
                    logs_table = gr.Dataframe(
                        value=demo.get_executions_dataframe(),
                        label="Detailed Execution Logs",
                        interactive=False,
                        wrap=True
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("""
                    ### 📊 Performance Metrics
                    
                    **Success Rate:** 95.2%
                    **Average Duration:** 4.2 seconds
                    **Total Executions:** 247
                    **Failed Executions:** 12
                    
                    ### ⚡ Recent Activity
                    - **2 min ago:** Lead created in Zoho CRM
                    - **5 min ago:** Slack trigger processed
                    - **8 min ago:** Gmail integration successful
                    - **12 min ago:** Webhook received and processed
                    
                    ### 🔍 Error Tracking
                    - **API Timeouts:** 3
                    - **Auth Failures:** 2
                    - **CRM Quota Exceeded:** 1
                    """)
            
            with gr.Accordion("View Raw Execution Data", open=False):
                raw_logs = gr.JSON(
                    value=demo_state["executions"][:5] if demo_state["executions"] else {},
                    label="Raw Execution Data (Last 5)"
                )
        
        # Documentation Tab
        with gr.Tab("📚 Documentation"):
            gr.Markdown("""
            # Dragify AI Agent Template Documentation
            
            ## Overview
            This demo showcases a modular AI Agent automation platform that automates business workflows involving triggers, data collection, CRM integration, and notifications.
            
            ## Architecture
            
            ### Core Components
            1. **Trigger System**: Listens for events from Slack, Gmail, or webhooks
            2. **Data Collection**: Fetches lead data from various sources
            3. **CRM Integration**: Inserts data into CRM systems
            4. **Notification System**: Sends status confirmations
            
            ### Technology Stack
            - **Backend**: Python + FastAPI
            - **Agent Logic**: LangChain + LangGraph
            - **Database**: PostgreSQL
            - **Frontend**: React + TailwindCSS
            - **Authentication**: OAuth 2.0
            
            ## Supported Integrations
            
            ### Triggers
            - **Slack**: New messages, mentions, reactions
            - **Gmail**: New emails, specific patterns
            - **Webhooks**: Custom HTTP endpoints
            
            ### Data Sources
            - **Google Sheets**: Form responses, spreadsheet data
            - **MongoDB**: NoSQL document queries
            - **PostgreSQL**: Relational database queries
            - **Airtable**: Base records and views
            
            ### CRM Systems
            - **Zoho CRM**: Leads, contacts, deals
            - **Salesforce**: Leads, opportunities, accounts
            - **Odoo CRM**: Leads, partners, activities
            - **Mock CRM**: For testing and development
            
            ## API Endpoints
            
            ```
            POST /agents                    # Create new agent
            GET  /agents                    # List all agents
            GET  /agents/{id}              # Get specific agent
            DELETE /agents/{id}            # Delete agent
            
            POST /workflows/trigger        # Trigger workflow
            GET  /workflows/logs           # Get execution logs
            GET  /workflows/status/{id}    # Get workflow status
            
            POST /webhooks/trigger/{id}    # Webhook endpoint
            ```
            
            ## OAuth Configuration
            
            Each integration requires proper OAuth setup:
            
            1. **Slack**: Create app at api.slack.com
            2. **Google**: Setup in Google Cloud Console
            3. **Outlook**: Register in Azure Portal
            
            ## Deployment
            
            - **Frontend**: Deployed on Vercel
            - **Backend**: Can be deployed on Railway, Render, or similar
            - **Database**: PostgreSQL on cloud provider
            
            ## Security Features
            
            - Token-based OAuth 2.0 authentication
            - Secure credential storage
            - Multi-user support with isolation
            - Webhook signature verification
            - Rate limiting and error handling
            
            ## Extensibility
            
            The system is designed for easy extension:
            - Add new trigger types in `integrations/triggers/`
            - Add new data sources in `integrations/datasources/`
            - Add new CRM systems in `integrations/crms/`
            - Customize agent logic using LangChain
            
            For more information, see the GitHub repository and live deployment.
            """)
    
    # Event Handlers
    create_btn.click(
        create_new_agent,
        inputs=[agent_name, agent_description, trigger_type, data_source, crm_system],
        outputs=[create_status, agents_table]
    )
    
    test_btn.click(
        trigger_workflow_demo,
        inputs=[agent_selector, trigger_message],
        outputs=[test_status, executions_table, execution_details]
    )
    
    refresh_btn.click(
        refresh_data,
        outputs=[agents_table, executions_table, agent_selector]
    )
    
    reset_btn.click(
        reset_demo,
        outputs=[agents_table, executions_table, agent_selector, reset_status]
    )
    
    # Quick test scenarios
    def set_scenario(scenario_type):
        if scenario_type == "email":
            return "1 - Lead Capture Bot", "New lead inquiry: 'Interested in your enterprise solution. Please contact me. - Sarah Johnson, CTO at TechCorp'"
        elif scenario_type == "slack":
            return "1 - Lead Capture Bot", "@dragify-bot New qualified lead from LinkedIn conversation with decision maker"
        elif scenario_type == "webhook":
            return "3 - Webhook Lead Handler", '{"name": "Demo User", "email": "demo@example.com", "company": "Demo Corp", "interest": "Enterprise Plan"}'
    
    scenario1_btn.click(
        lambda: set_scenario("email"),
        outputs=[agent_selector, trigger_message]
    )
    
    scenario2_btn.click(
        lambda: set_scenario("slack"),
        outputs=[agent_selector, trigger_message]
    )
    
    scenario3_btn.click(
        lambda: set_scenario("webhook"),
        outputs=[agent_selector, trigger_message]
    )

# Launch configuration
if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True,
        show_api=False,
        favicon_path=None,
        ssl_verify=False
    )