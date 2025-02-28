#!/bin/bash

# Colors for output
CYAN='\033[1;36m'    
RED='\033[1;31m'     
GREEN='\033[1;32m'   
YELLOW='\033[1;33m'
NC='\033[0m'         

PORT=8000
LOG_FILE="/tmp/tts_daemon.log"
MAX_WAIT=30

# Check for existing processes
cleanup_existing() {
    echo -e "${CYAN}Checking for existing processes...${NC}"
    
    # Check for running API processes
    if pgrep -f "python3 api.py" > /dev/null; then
        echo -e "${YELLOW}Found existing API processes. Cleaning up...${NC}"
        pkill -f "python3 api.py"
        sleep 2
    fi
    
    # Check if port is in use
    if netstat -tuln | grep -q ":$PORT "; then
        echo -e "${RED}Port $PORT is still in use. Please run cleanup.sh first${NC}"
        return 1
    fi
    
    return 0
}

# Check if API is already running
check_api_running() {
    if curl -s "http://localhost:$PORT/health" > /dev/null; then
        echo -e "${YELLOW}API is already running on port $PORT${NC}"
        return 0
    fi
    return 1
}

# Start the API server
start_api() {
    echo -e "${CYAN}Starting API server...${NC}"
    
    # Check if api.py exists
    if [ ! -f "api.py" ]; then
        echo -e "${RED}Error: api.py not found in current directory: $(pwd)${NC}"
        return 1
    fi
    
    # Remove old log file if it exists
    rm -f "$LOG_FILE"
    
    # Start the server with output to both terminal and log file
    echo -e "${CYAN}Starting Python process...${NC}"
    python3 api.py 2>&1 | tee "$LOG_FILE" &
    local API_PID=$!
    
    echo -e "${CYAN}API server starting with PID: ${PURPLE}$API_PID${NC}"
    echo -e "${CYAN}Waiting for server to be ready (max ${MAX_WAIT}s)...${NC}"
    
    # Wait for server to be ready
    for i in $(seq 1 $MAX_WAIT); do
        echo -n "."
        
        # Check if process is still running
        if ! kill -0 $API_PID 2>/dev/null; then
            echo -e "\n${RED}Server process died. Recent log entries:${NC}"
            if [ -f "$LOG_FILE" ]; then
                tail -n 10 "$LOG_FILE"
            else
                echo -e "${RED}No log file found at $LOG_FILE${NC}"
            fi
            return 1
        fi
        
        # Check if server is responding
        if curl -s "http://localhost:$PORT/health" > /dev/null; then
            echo -e "\n${GREEN}Server is ready!${NC}"
            return 0
        fi
        
        # Show recent log entries if available
        if [ -f "$LOG_FILE" ]; then
            tail -n 1 "$LOG_FILE"
        fi
        
        sleep 1
    done
    
    echo -e "\n${RED}Server failed to start within ${MAX_WAIT} seconds${NC}"
    if [ -f "$LOG_FILE" ]; then
        echo -e "${YELLOW}Recent log entries:${NC}"
        tail -n 20 "$LOG_FILE"
    fi
    return 1
}

# Main execution
if ! cleanup_existing; then
    exit 1
fi

if check_api_running; then
    echo -e "${GREEN}API server is already running${NC}"
    exit 0
fi

if start_api; then
    echo -e "${GREEN}API server started successfully${NC}"
    exit 0
else
    echo -e "${RED}Failed to start API server${NC}"
    exit 1
fi
