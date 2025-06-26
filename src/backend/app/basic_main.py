"""
Dragify AI Agent Template - Enhanced Main Application
Comprehensive modular AI agent automation platform with integrated Gradio demo
"""

import os
import sys
import logging
import asyncio
import threading
import argparse
import uvicorn
import gradio as gr
import pandas as pd
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from contextlib import asynccontextmanager
from pathlib import Path
import random

# FastAPI and related imports
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import jwt
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dragify.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dragify.db")
    JWT_SECRET = os.getenv("JWT_SECRET_KEY", "dragify-secret-key-2024")
    JWT_ALGORITHM = "HS256"
    GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
    API_PORT = int(os.getenv("API_PORT", "8000"))
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Database setup
engine = create_engine(Config.DATABASE_URL, echo=Config.ENVIRONMENT == "development")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Security
security = HTTPBearer()

# Enums for configuration
class TriggerType(str, Enum):
    SLACK = "slack"
    GMAIL = "gmail"
    WEBHOOK = "webhook"
    TELEGRAM = "telegram"
    DISCORD = "discord"

class DataSourceType(str, Enum):
    GOOGLE_SHEETS = "google_sheets"
    MONGODB = "mongodb"
    POSTGRESQL = "postgresql"
    AIRTABLE = "airtable"
    WEBHOOK = "webhook"
    CSV_FILE = "csv_file"
    JSON_API = "json_api"

class CRMType(str, Enum):
    ZOHO = "zoho"
    SALESFORCE = "salesforce"
    ODOO = "odoo"
    HUBSPOT = "hubspot"
    PIPEDRIVE = "pipedrive"
    MOCK = "mock"

# Enhanced Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    subscription_tier = Column(String, default="free")

class OAuthToken(Base):
    __tablename__ = "oauth_tokens"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    provider = Column(String)
    access_token = Column(Text)
    refresh_token = Column(Text, nullable=True)
    expires_at = Column(DateTime)
    scope = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class Agent(Base):
    __tablename__ = "agents"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    name = Column(String)
    description = Column(Text, nullable=True)
    trigger_type = Column(String)
    trigger_config = Column(JSON)
    data_source_type = Column(String)
    data_source_config = Column(JSON)
    crm_type = Column(String)
    crm_config = Column(JSON)
    notification_config = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True)
    execution_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class WorkflowExecution(Base):
    __tablename__ = "workflow_executions"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, index=True)
    trigger_data = Column(JSON)
    fetched_data = Column(JSON, nullable=True)
    crm_response = Column(JSON, nullable=True)
    notification_response = Column(JSON, nullable=True)
    status = Column(String)  # pending, success, failed, retry
    error_message = Column(Text, nullable=True)
    execution_time_ms = Column(Integer, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    retry_count = Column(Integer, default=0)

class ApiKey(Base):
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    key_name = Column(String)
    key_hash = Column(String)
    permissions = Column(JSON)
    is_active = Column(Boolean, default=True)
    last_used = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)

# Create tables
Base.metadata.create_all(bind=engine)

# Enhanced Pydantic Models
class AgentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    trigger_type: TriggerType
    trigger_config: dict
    data_source_type: DataSourceType
    data_source_config: dict
    crm_type: CRMType
    crm_config: dict
    notification_config: Optional[dict] = None

class AgentUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    is_active: Optional[bool] = None
    trigger_config: Optional[dict] = None
    data_source_config: Optional[dict] = None
    crm_config: Optional[dict] = None
    notification_config: Optional[dict] = None

class AgentResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    trigger_type: str
    data_source_type: str
    crm_type: str
    is_active: bool
    execution_count: int
    success_count: int
    success_rate: float
    created_at: datetime
    updated_at: datetime

class WorkflowExecutionResponse(BaseModel):
    id: int
    agent_id: int
    status: str
    execution_time_ms: Optional[int]
    started_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]
    retry_count: int

class TriggerData(BaseModel):
    agent_id: int
    data: dict
    priority: Optional[str] = "normal"  # low, normal, high
    async_execution: Optional[bool] = True

class StatsResponse(BaseModel):
    total_agents: int
    active_agents: int
    total_executions: int
    successful_executions: int
    failed_executions: int
    success_rate: float
    avg_execution_time: float

# Enhanced AI Agent Class
class DragifyAgent:
    def __init__(self, agent_config: Agent):
        self.agent_config = agent_config
        self.trigger_type = agent_config.trigger_type
        self.data_source_type = agent_config.data_source_type
        self.crm_type = agent_config.crm_type
        self.logger = logging.getLogger(f"agent_{agent_config.id}")
    
    async def execute_workflow(self, trigger_data: dict, db: Session) -> WorkflowExecution:
        """Execute the complete workflow with enhanced error handling and logging"""
        start_time = time.time()
        execution = WorkflowExecution(
            agent_id=self.agent_config.id,
            trigger_data=trigger_data,
            status="pending"
        )
        db.add(execution)
        db.commit()
        db.refresh(execution)
        
        try:
            self.logger.info(f"Starting workflow execution {execution.id}")
            
            # Step 1: Validate trigger data
            await self._validate_trigger_data(trigger_data)
            
            # Step 2: Fetch lead data from configured source
            self.logger.info("Fetching lead data from data source")
            lead_data = await self._fetch_lead_data(trigger_data)
            execution.fetched_data = lead_data
            db.commit()
            
            # Step 3: Enrich data if configured
            if self.agent_config.data_source_config.get("enable_enrichment"):
                lead_data = await self._enrich_lead_data(lead_data)
            
            # Step 4: Insert into CRM with retry logic
            self.logger.info("Inserting data into CRM")
            crm_response = await self._insert_to_crm_with_retry(lead_data)
            execution.crm_response = crm_response
            
            # Step 5: Send notifications
            notification_response = await self._send_notifications(lead_data, crm_response, True)
            execution.notification_response = notification_response
            
            # Update execution status
            execution.status = "success"
            execution.completed_at = datetime.utcnow()
            execution.execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Update agent statistics
            self.agent_config.execution_count += 1
            self.agent_config.success_count += 1
            
            self.logger.info(f"Workflow execution {execution.id} completed successfully")
            
        except Exception as e:
            self.logger.error(f"Workflow execution {execution.id} failed: {str(e)}")
            execution.status = "failed"
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()
            execution.execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Update agent statistics
            self.agent_config.execution_count += 1
            
            # Send failure notification
            await self._send_notifications(
                execution.fetched_data or {}, 
                {}, 
                False, 
                str(e)
            )
        
        db.commit()
        db.refresh(execution)
        return execution
    
    async def _validate_trigger_data(self, trigger_data: dict):
        """Validate incoming trigger data"""
        required_fields = self.agent_config.trigger_config.get("required_fields", [])
        for field in required_fields:
            if field not in trigger_data:
                raise ValueError(f"Required field '{field}' missing from trigger data")
    
    async def _fetch_lead_data(self, trigger_data: dict) -> dict:
        """Enhanced lead data fetching with multiple source support"""
        config = self.agent_config.data_source_config
        
        try:
            if self.data_source_type == DataSourceType.GOOGLE_SHEETS:
                return await self._fetch_from_google_sheets(config, trigger_data)
            elif self.data_source_type == DataSourceType.MONGODB:
                return await self._fetch_from_mongodb(config, trigger_data)
            elif self.data_source_type == DataSourceType.POSTGRESQL:
                return await self._fetch_from_postgresql(config, trigger_data)
            elif self.data_source_type == DataSourceType.AIRTABLE:
                return await self._fetch_from_airtable(config, trigger_data)
            elif self.data_source_type == DataSourceType.WEBHOOK:
                return trigger_data
            elif self.data_source_type == DataSourceType.CSV_FILE:
                return await self._fetch_from_csv(config, trigger_data)
            elif self.data_source_type == DataSourceType.JSON_API:
                return await self._fetch_from_json_api(config, trigger_data)
            else:
                raise ValueError(f"Unsupported data source: {self.data_source_type}")
        except Exception as e:
            self.logger.error(f"Failed to fetch data from {self.data_source_type}: {str(e)}")
            raise
    
    async def _fetch_from_google_sheets(self, config: dict, trigger_data: dict) -> dict:
        """Fetch data from Google Sheets with authentication"""
        # Mock implementation with more realistic data
        await asyncio.sleep(0.5)  # Simulate API call
        return {
            "name": f"Lead_{random.randint(1000, 9999)}",
            "email": f"lead{random.randint(100, 999)}@example.com",
            "company": random.choice(["TechCorp", "InnovateLLC", "StartupXYZ", "Enterprise Inc"]),
            "phone": f"+1-555-{random.randint(1000, 9999)}",
            "source": "google_sheets",
            "lead_score": random.randint(50, 100),
            "interest_level": random.choice(["high", "medium", "low"]),
            "sheet_id": config.get("sheet_id"),
            "row_id": random.randint(1, 1000),
            "trigger_info": trigger_data,
            "fetched_at": datetime.utcnow().isoformat()
        }
    
    async def _fetch_from_json_api(self, config: dict, trigger_data: dict) -> dict:
        """Fetch data from JSON API endpoint"""
        await asyncio.sleep(0.3)
        return {
            "name": "API Lead",
            "email": "api@example.com",
            "company": "API Corp",
            "phone": "+1-555-API1",
            "source": "json_api",
            "api_endpoint": config.get("endpoint"),
            "trigger_info": trigger_data,
            "fetched_at": datetime.utcnow().isoformat()
        }
    
    async def _fetch_from_csv(self, config: dict, trigger_data: dict) -> dict:
        """Fetch data from CSV file"""
        await asyncio.sleep(0.2)
        return {
            "name": "CSV Lead",
            "email": "csv@example.com",
            "company": "CSV Corp",
            "phone": "+1-555-CSV1",
            "source": "csv_file",
            "file_path": config.get("file_path"),
            "trigger_info": trigger_data,
            "fetched_at": datetime.utcnow().isoformat()
        }
    
    async def _enrich_lead_data(self, lead_data: dict) -> dict:
        """Enrich lead data with additional information"""
        # Mock enrichment service
        lead_data["enriched"] = True
        lead_data["company_size"] = random.choice(["1-10", "11-50", "51-200", "201-1000", "1000+"])
        lead_data["industry"] = random.choice(["Technology", "Healthcare", "Finance", "Manufacturing", "Retail"])
        lead_data["revenue_estimate"] = f"${random.randint(1, 100)}M"
        return lead_data
    
    async def _insert_to_crm_with_retry(self, lead_data: dict, max_retries: int = 3) -> dict:
        """Insert to CRM with retry logic"""
        for attempt in range(max_retries):
            try:
                return await self._insert_to_crm(lead_data)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                self.logger.warning(f"CRM insertion attempt {attempt + 1} failed, retrying...")
    
    async def _insert_to_crm(self, lead_data: dict) -> dict:
        """Enhanced CRM insertion with multiple CRM support"""
        config = self.agent_config.crm_config
        
        try:
            if self.crm_type == CRMType.ZOHO:
                return await self._insert_to_zoho(config, lead_data)
            elif self.crm_type == CRMType.SALESFORCE:
                return await self._insert_to_salesforce(config, lead_data)
            elif self.crm_type == CRMType.ODOO:
                return await self._insert_to_odoo(config, lead_data)
            elif self.crm_type == CRMType.HUBSPOT:
                return await self._insert_to_hubspot(config, lead_data)
            elif self.crm_type == CRMType.PIPEDRIVE:
                return await self._insert_to_pipedrive(config, lead_data)
            elif self.crm_type == CRMType.MOCK:
                return await self._insert_to_mock_crm(config, lead_data)
            else:
                raise ValueError(f"Unsupported CRM: {self.crm_type}")
        except Exception as e:
            self.logger.error(f"Failed to insert to {self.crm_type}: {str(e)}")
            raise
    
    async def _insert_to_hubspot(self, config: dict, lead_data: dict) -> dict:
        """Insert lead into HubSpot CRM"""
        await asyncio.sleep(0.4)
        success = random.random() > 0.05  # 95% success rate
        if not success:
            raise Exception("HubSpot API quota exceeded")
        
        return {
            "crm": "hubspot",
            "lead_id": f"HS_{uuid.uuid4().hex[:8].upper()}",
            "contact_id": f"CONTACT_{random.randint(100000, 999999)}",
            "status": "created",
            "message": "Lead successfully created in HubSpot",
            "pipeline_stage": "New Lead",
            "owner_id": config.get("default_owner_id", "auto_assign")
        }
    
    async def _insert_to_pipedrive(self, config: dict, lead_data: dict) -> dict:
        """Insert lead into Pipedrive CRM"""
        await asyncio.sleep(0.3)
        return {
            "crm": "pipedrive",
            "lead_id": f"PD_{uuid.uuid4().hex[:8].upper()}",
            "person_id": random.randint(1000, 9999),
            "organization_id": random.randint(100, 999),
            "status": "created",
            "message": "Lead successfully created in Pipedrive",
            "pipeline_id": config.get("pipeline_id", 1),
            "stage_id": config.get("stage_id", 1)
        }
    
    async def _send_notifications(self, lead_data: dict, crm_response: dict, success: bool, error: str = None) -> dict:
        """Enhanced notification system with multiple channels"""
        notification_config = self.agent_config.notification_config or {}
        notifications_sent = []
        
        try:
            # Email notification
            if notification_config.get("email", {}).get("enabled", True):
                email_result = await self._send_email_notification(lead_data, crm_response, success, error)
                notifications_sent.append(email_result)
            
            # Slack notification
            if notification_config.get("slack", {}).get("enabled", False):
                slack_result = await self._send_slack_notification(lead_data, crm_response, success, error)
                notifications_sent.append(slack_result)
            
            # Webhook notification
            if notification_config.get("webhook", {}).get("enabled", False):
                webhook_result = await self._send_webhook_notification(lead_data, crm_response, success, error)
                notifications_sent.append(webhook_result)
            
            return {
                "notifications_sent": len(notifications_sent),
                "results": notifications_sent,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to send notifications: {str(e)}")
            return {
                "notifications_sent": 0,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _send_email_notification(self, lead_data: dict, crm_response: dict, success: bool, error: str = None) -> dict:
        """Send email notification"""
        await asyncio.sleep(0.1)  # Simulate email sending
        status = "SUCCESS" if success else "FAILED"
        return {
            "channel": "email",
            "status": "sent",
            "recipient": "admin@company.com",
            "subject": f"Workflow {status} - Agent {self.agent_config.name}",
            "sent_at": datetime.utcnow().isoformat()
        }
    
    async def _send_slack_notification(self, lead_data: dict, crm_response: dict, success: bool, error: str = None) -> dict:
        """Send Slack notification"""
        await asyncio.sleep(0.2)
        return {
            "channel": "slack",
            "status": "sent",
            "channel_id": "#leads",
            "message_ts": str(time.time()),
            "sent_at": datetime.utcnow().isoformat()
        }
    
    async def _send_webhook_notification(self, lead_data: dict, crm_response: dict, success: bool, error: str = None) -> dict:
        """Send webhook notification"""
        await asyncio.sleep(0.1)
        return {
            "channel": "webhook",
            "status": "sent",
            "endpoint": "https://api.company.com/notifications",
            "response_code": 200,
            "sent_at": datetime.utcnow().isoformat()
        }

# Sample data for Gradio demo
SAMPLE_LEADS = [
    {"name": "John Doe", "email": "john@acme.com", "company": "Acme Corp", "phone": "+1-555-0123", "source": "web_form"},
    {"name": "Jane Smith", "email": "jane@techsol.com", "company": "Tech Solutions", "phone": "+1-555-0456", "source": "linkedin"},
    {"name": "Bob Johnson", "email": "bob@enterprise.com", "company": "Enterprise LLC", "phone": "+1-555-0789", "source": "referral"},
    {"name": "Alice Brown", "email": "alice@startup.com", "company": "StartupXYZ", "phone": "+1-555-0101", "source": "event"},
    {"name": "Charlie Wilson", "email": "charlie@innovate.com", "company": "Innovate Inc", "phone": "+1-555-0202", "source": "cold_email"},
]

# Demo state for Gradio
demo_state = {
    "agents": [],
    "executions": [],
    "agent_counter": 0
}

class GradioDemo:
    """Enhanced Gradio Demo Interface"""
    
    def __init__(self):
        self.reset_demo_data()
    
    def reset_demo_data(self):
        """Reset demo data to initial state"""
        demo_state["agents"] = []
        demo_state["executions"] = []
        demo_state["agent_counter"] = 0
        self.create_sample_agents()
    
    def create_sample_agents(self):
        """Create comprehensive sample agents"""
        sample_agents = [
            {
                "name": "Lead Capture Bot",
                "description": "Captures leads from Slack mentions and adds to CRM with enrichment",
                "trigger": "Slack Message",
                "data_source": "Google Sheets",
                "crm": "Zoho CRM",
                "status": "Active"
            },
            {
                "name": "Email Lead Processor",
                "description": "Processes leads from Gmail and syncs with Salesforce pipeline",
                "trigger": "Gmail Email",
                "data_source": "MongoDB",
                "crm": "Salesforce",
                "status": "Active"
            },
            {
                "name": "Webhook Lead Handler",
                "description": "Handles web form submissions via webhook with auto-assignment",
                "trigger": "Webhook",
                "data_source": "Airtable",
                "crm": "HubSpot",
                "status": "Active"
            },
            {
                "name": "CSV Import Agent",
                "description": "Processes CSV uploads and imports to CRM with validation",
                "trigger": "Webhook",
                "data_source": "CSV File",
                "crm": "Pipedrive",
                "status": "Active"
            },
            {
                "name": "API Lead Sync",
                "description": "Syncs leads from external API to CRM with deduplication",
                "trigger": "Webhook",
                "data_source": "JSON API",
                "crm": "Odoo CRM",
                "status": "Active"
            }
        ]
        
        for agent_data in sample_agents:
            self.create_agent(
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
            "executions": random.randint(10, 100),
            "success_rate": f"{random.randint(85, 100)}%",
            "avg_response_time": f"{random.uniform(1.5, 5.0):.1f}s",
            "last_execution": (datetime.now() - timedelta(minutes=random.randint(5, 180))).strftime("%Y-%m-%d %H:%M:%S")
        }
        
        demo_state["agents"].append(agent)
        return agent_id
    
    def get_agents_dataframe(self) -> pd.DataFrame:
        """Get agents as DataFrame for display"""
        if not demo_state["agents"]:
            return pd.DataFrame(columns=["ID", "Name", "Trigger", "Data Source", "CRM", "Status", "Executions", "Success Rate", "Avg Time"])
        
        agents_data = []
        for agent in demo_state["agents"]:
            agents_data.append([
                agent["id"],
                agent["name"],
                agent["trigger"],
                agent["data_source"],
                agent["crm"],
                agent["status"],
                agent["executions"],
                agent["success_rate"],
                agent["avg_response_time"]
            ])
        
        return pd.DataFrame(agents_data, columns=["ID", "Name", "Trigger", "Data Source", "CRM", "Status", "Executions", "Success Rate", "Avg Time"])
    
    def execute_workflow(self, agent_id: int, trigger_data: Dict) -> Dict:
        """Execute a workflow for testing with enhanced response"""
        agent = next((a for a in demo_state["agents"] if a["id"] == agent_id), None)
        if not agent:
            return {"error": "Agent not found"}
        
        execution_id = len(demo_state["executions"]) + 1
        
        # Simulate more realistic workflow execution
        lead_data = random.choice(SAMPLE_LEADS).copy()
        lead_data.update({
            "trigger_source": agent["trigger"],
            "fetched_from": agent["data_source"],
            "lead_score": random.randint(50, 100),
            "enriched": True,
            "company_size": random.choice(["1-10", "11-50", "51-200", "201-1000", "1000+"]),
            "industry": random.choice(["Technology", "Healthcare", "Finance", "Manufacturing"])
        })
        
        # Simulate CRM insertion with more details
        success = random.random() > 0.1  # 90% success rate
        crm_response = {
            "crm": agent["crm"],
            "lead_id": f"{agent['crm'].upper().replace(' ', '_')}_{uuid.uuid4().hex[:8]}",
            "contact_id": f"CONTACT_{random.randint(100000, 999999)}",
            "status": "created" if success else "failed",
            "pipeline_stage": "New Lead" if success else None,
            "owner_assigned": "auto" if success else None,
            "timestamp": datetime.now().isoformat(),
            "processing_time": f"{random.uniform(0.5, 3.0):.2f}s"
        }
        
        # Simulate notifications
        notifications = {
            "email_sent": True,
            "slack_notified": random.choice([True, False]),
            "webhook_called": random.choice([True, False]),
            "notification_count": random.randint(1, 3)
        }
        
        execution = {
            "id": execution_id,
            "agent_id": agent_id,
            "agent_name": agent["name"],
            "trigger_data": trigger_data,
            "lead_data": lead_data,
            "crm_response": crm_response,
            "notifications": notifications,
            "status": "Success" if success else "Failed",
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration": f"{random.uniform(2.0, 8.0):.1f} seconds",
            "execution_steps": [