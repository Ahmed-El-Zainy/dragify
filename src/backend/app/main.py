"""
Dragify AI Agent Template - Main FastAPI Application
"""
import os
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timedelta
import jwt
import asyncio
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dragify.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Security
security = HTTPBearer()
JWT_SECRET = os.getenv("JWT_SECRET_KEY", "your-secret-key")
JWT_ALGORITHM = "HS256"

# Enums for configuration
class TriggerType(str, Enum):
    SLACK = "slack"
    GMAIL = "gmail"
    WEBHOOK = "webhook"

class DataSourceType(str, Enum):
    GOOGLE_SHEETS = "google_sheets"
    MONGODB = "mongodb"
    POSTGRESQL = "postgresql"
    AIRTABLE = "airtable"
    WEBHOOK = "webhook"

class CRMType(str, Enum):
    ZOHO = "zoho"
    SALESFORCE = "salesforce"
    ODOO = "odoo"
    MOCK = "mock"

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class OAuthToken(Base):
    __tablename__ = "oauth_tokens"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    provider = Column(String)  # slack, google, outlook
    access_token = Column(String)
    refresh_token = Column(String, nullable=True)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

class Agent(Base):
    __tablename__ = "agents"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    name = Column(String)
    description = Column(String, nullable=True)
    trigger_type = Column(String)
    trigger_config = Column(JSON)
    data_source_type = Column(String)
    data_source_config = Column(JSON)
    crm_type = Column(String)
    crm_config = Column(JSON)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class WorkflowExecution(Base):
    __tablename__ = "workflow_executions"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, index=True)
    trigger_data = Column(JSON)
    fetched_data = Column(JSON, nullable=True)
    crm_response = Column(JSON, nullable=True)
    status = Column(String)  # pending, success, failed, retry
    error_message = Column(String, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    retry_count = Column(Integer, default=0)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic Models
class AgentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    trigger_type: TriggerType
    trigger_config: dict
    data_source_type: DataSourceType
    data_source_config: dict
    crm_type: CRMType
    crm_config: dict

class AgentResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    trigger_type: str
    data_source_type: str
    crm_type: str
    is_active: bool
    created_at: datetime

class WorkflowExecutionResponse(BaseModel):
    id: int
    agent_id: int
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]
    retry_count: int

class TriggerData(BaseModel):
    agent_id: int
    data: dict

# AI Agent Class using LangChain concepts
class DragifyAgent:
    def __init__(self, agent_config: Agent):
        self.agent_config = agent_config
        self.trigger_type = agent_config.trigger_type
        self.data_source_type = agent_config.data_source_type
        self.crm_type = agent_config.crm_type
    
    async def execute_workflow(self, trigger_data: dict, db: Session) -> WorkflowExecution:
        """Execute the complete workflow: trigger -> data collection -> CRM -> notification"""
        execution = WorkflowExecution(
            agent_id=self.agent_config.id,
            trigger_data=trigger_data,
            status="pending"
        )
        db.add(execution)
        db.commit()
        db.refresh(execution)
        
        try:
            # Step 1: Process trigger data
            logger.info(f"Processing trigger data for agent {self.agent_config.id}")
            
            # Step 2: Fetch lead data from configured source
            lead_data = await self._fetch_lead_data(trigger_data)
            execution.fetched_data = lead_data
            
            # Step 3: Insert into CRM
            crm_response = await self._insert_to_crm(lead_data)
            execution.crm_response = crm_response
            
            # Step 4: Send notification
            await self._send_notification(lead_data, crm_response, True)
            
            execution.status = "success"
            execution.completed_at = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            execution.status = "failed"
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()
            
            # Send failure notification
            await self._send_notification(
                execution.fetched_data or {}, 
                {}, 
                False, 
                str(e)
            )
        
        db.commit()
        db.refresh(execution)
        return execution
    
    async def _fetch_lead_data(self, trigger_data: dict) -> dict:
        """Fetch lead data from configured data source"""
        config = self.agent_config.data_source_config
        
        if self.data_source_type == DataSourceType.GOOGLE_SHEETS:
            return await self._fetch_from_google_sheets(config, trigger_data)
        elif self.data_source_type == DataSourceType.MONGODB:
            return await self._fetch_from_mongodb(config, trigger_data)
        elif self.data_source_type == DataSourceType.POSTGRESQL:
            return await self._fetch_from_postgresql(config, trigger_data)
        elif self.data_source_type == DataSourceType.AIRTABLE:
            return await self._fetch_from_airtable(config, trigger_data)
        elif self.data_source_type == DataSourceType.WEBHOOK:
            return trigger_data  # Data comes directly from webhook
        else:
            raise ValueError(f"Unsupported data source: {self.data_source_type}")
    
    async def _fetch_from_google_sheets(self, config: dict, trigger_data: dict) -> dict:
        """Fetch data from Google Sheets"""
        # Mock implementation - replace with actual Google Sheets API
        return {
            "name": "John Doe",
            "email": "john@example.com",
            "company": "Acme Corp",
            "phone": "+1-555-0123",
            "source": "google_sheets",
            "sheet_id": config.get("sheet_id"),
            "trigger_info": trigger_data
        }
    
    async def _fetch_from_mongodb(self, config: dict, trigger_data: dict) -> dict:
        """Fetch data from MongoDB"""
        # Mock implementation - replace with actual MongoDB queries
        return {
            "name": "Jane Smith",
            "email": "jane@example.com",
            "company": "Tech Solutions",
            "phone": "+1-555-0456",
            "source": "mongodb",
            "collection": config.get("collection"),
            "trigger_info": trigger_data
        }
    
    async def _fetch_from_postgresql(self, config: dict, trigger_data: dict) -> dict:
        """Fetch data from PostgreSQL"""
        # Mock implementation - replace with actual PostgreSQL queries
        return {
            "name": "Bob Johnson",
            "email": "bob@example.com",
            "company": "Enterprise LLC",
            "phone": "+1-555-0789",
            "source": "postgresql",
            "table": config.get("table"),
            "trigger_info": trigger_data
        }
    
    async def _fetch_from_airtable(self, config: dict, trigger_data: dict) -> dict:
        """Fetch data from Airtable"""
        # Mock implementation - replace with actual Airtable API
        return {
            "name": "Alice Brown",
            "email": "alice@example.com",
            "company": "StartupXYZ",
            "phone": "+1-555-0101",
            "source": "airtable",
            "base_id": config.get("base_id"),
            "trigger_info": trigger_data
        }
    
    async def _insert_to_crm(self, lead_data: dict) -> dict:
        """Insert lead data into configured CRM"""
        config = self.agent_config.crm_config
        
        if self.crm_type == CRMType.ZOHO:
            return await self._insert_to_zoho(config, lead_data)
        elif self.crm_type == CRMType.SALESFORCE:
            return await self._insert_to_salesforce(config, lead_data)
        elif self.crm_type == CRMType.ODOO:
            return await self._insert_to_odoo(config, lead_data)
        elif self.crm_type == CRMType.MOCK:
            return await self._insert_to_mock_crm(config, lead_data)
        else:
            raise ValueError(f"Unsupported CRM: {self.crm_type}")
    
    async def _insert_to_zoho(self, config: dict, lead_data: dict) -> dict:
        """Insert lead into Zoho CRM"""
        # Mock implementation - replace with actual Zoho CRM API
        return {
            "crm": "zoho",
            "lead_id": "ZOHO_123456",
            "status": "created",
            "message": "Lead successfully created in Zoho CRM"
        }
    
    async def _insert_to_salesforce(self, config: dict, lead_data: dict) -> dict:
        """Insert lead into Salesforce"""
        # Mock implementation - replace with actual Salesforce API
        return {
            "crm": "salesforce",
            "lead_id": "SF_789012",
            "status": "created",
            "message": "Lead successfully created in Salesforce"
        }
    
    async def _insert_to_odoo(self, config: dict, lead_data: dict) -> dict:
        """Insert lead into Odoo CRM"""
        # Mock implementation - replace with actual Odoo API
        return {
            "crm": "odoo",
            "lead_id": "ODOO_345678",
            "status": "created",
            "message": "Lead successfully created in Odoo CRM"
        }
    
    async def _insert_to_mock_crm(self, config: dict, lead_data: dict) -> dict:
        """Insert lead into mock CRM for testing"""
        return {
            "crm": "mock",
            "lead_id": f"MOCK_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "status": "created",
            "message": "Lead successfully created in Mock CRM",
            "lead_data": lead_data
        }
    
    async def _send_notification(self, lead_data: dict, crm_response: dict, success: bool, error: str = None):
        """Send email notification about workflow execution"""
        # Mock implementation - replace with actual email service
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"Notification sent: Workflow {status} for agent {self.agent_config.id}")
        
        if not success and error:
            logger.error(f"Error details: {error}")

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Dependency to get current user (simplified for demo)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except jwt.PyJWTError:
        # For demo purposes, return a default user
        return 1

# Create FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Dragify AI Agent Template")
    yield
    # Shutdown
    logger.info("Shutting down Dragify AI Agent Template")

app = FastAPI(
    title="Dragify AI Agent Template",
    description="Modular AI Agent automation platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://dragify-agent.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
@app.get("/")
async def root():
    return {"message": "Dragify AI Agent Template API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# Agent Management
@app.post("/agents", response_model=AgentResponse)
async def create_agent(
    agent_data: AgentCreate,
    db: Session = Depends(get_db),
    current_user: int = Depends(get_current_user)
):
    agent = Agent(
        user_id=current_user,
        name=agent_data.name,
        description=agent_data.description,
        trigger_type=agent_data.trigger_type.value,
        trigger_config=agent_data.trigger_config,
        data_source_type=agent_data.data_source_type.value,
        data_source_config=agent_data.data_source_config,
        crm_type=agent_data.crm_type.value,
        crm_config=agent_data.crm_config
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return agent

@app.get("/agents", response_model=List[AgentResponse])
async def list_agents(
    db: Session = Depends(get_db),
    current_user: int = Depends(get_current_user)
):
    agents = db.query(Agent).filter(Agent.user_id == current_user).all()
    return agents

@app.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: int,
    db: Session = Depends(get_db),
    current_user: int = Depends(get_current_user)
):
    agent = db.query(Agent).filter(
        Agent.id == agent_id,
        Agent.user_id == current_user
    ).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@app.delete("/agents/{agent_id}")
async def delete_agent(
    agent_id: int,
    db: Session = Depends(get_db),
    current_user: int = Depends(get_current_user)
):
    agent = db.query(Agent).filter(
        Agent.id == agent_id,
        Agent.user_id == current_user
    ).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    db.delete(agent)
    db.commit()
    return {"message": "Agent deleted successfully"}

# Workflow Execution
@app.post("/workflows/trigger")
async def trigger_workflow(
    trigger_data: TriggerData,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: int = Depends(get_current_user)
):
    # Get agent
    agent = db.query(Agent).filter(
        Agent.id == trigger_data.agent_id,
        Agent.user_id == current_user,
        Agent.is_active == True
    ).first()
    
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found or inactive")
    
    # Create and execute workflow
    dragify_agent = DragifyAgent(agent)
    execution = await dragify_agent.execute_workflow(trigger_data.data, db)
    
    return {"execution_id": execution.id, "status": execution.status}

@app.get("/workflows/logs", response_model=List[WorkflowExecutionResponse])
async def get_workflow_logs(
    agent_id: Optional[int] = None,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: int = Depends(get_current_user)
):
    query = db.query(WorkflowExecution).join(Agent).filter(Agent.user_id == current_user)
    
    if agent_id:
        query = query.filter(WorkflowExecution.agent_id == agent_id)
    
    executions = query.order_by(WorkflowExecution.started_at.desc()).limit(limit).all()
    return executions

@app.get("/workflows/status/{execution_id}")
async def get_workflow_status(
    execution_id: int,
    db: Session = Depends(get_db),
    current_user: int = Depends(get_current_user)
):
    execution = db.query(WorkflowExecution).join(Agent).filter(
        WorkflowExecution.id == execution_id,
        Agent.user_id == current_user
    ).first()
    
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    return {
        "id": execution.id,
        "status": execution.status,
        "trigger_data": execution.trigger_data,
        "fetched_data": execution.fetched_data,
        "crm_response": execution.crm_response,
        "started_at": execution.started_at,
        "completed_at": execution.completed_at,
        "error_message": execution.error_message,
        "retry_count": execution.retry_count
    }

# Mock CRM endpoints for testing
@app.post("/mock/zoho/leads")
async def mock_zoho_create_lead(lead_data: dict):
    return {
        "success": True,
        "lead_id": "ZOHO_MOCK_123",
        "message": "Lead created in Zoho CRM (Mock)"
    }

@app.post("/mock/salesforce/leads")
async def mock_salesforce_create_lead(lead_data: dict):
    return {
        "success": True,
        "lead_id": "SF_MOCK_456",
        "message": "Lead created in Salesforce (Mock)"
    }

@app.post("/mock/odoo/leads")
async def mock_odoo_create_lead(lead_data: dict):
    return {
        "success": True,
        "lead_id": "ODOO_MOCK_789",
        "message": "Lead created in Odoo CRM (Mock)"
    }

# Webhook endpoint for external triggers
@app.post("/webhooks/trigger/{agent_id}")
async def webhook_trigger(
    agent_id: int,
    webhook_data: dict,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    agent = db.query(Agent).filter(
        Agent.id == agent_id,
        Agent.trigger_type == "webhook",
        Agent.is_active == True
    ).first()
    
    if not agent:
        raise HTTPException(status_code=404, detail="Webhook agent not found")
    
    # Execute workflow in background
    dragify_agent = DragifyAgent(agent)
    execution = await dragify_agent.execute_workflow(webhook_data, db)
    
    return {"execution_id": execution.id, "status": "triggered"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8)