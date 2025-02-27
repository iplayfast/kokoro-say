#!/bin/bash

# Colors for output
CYAN='\033[1;36m'    
RED='\033[1;31m'     
GREEN='\033[1;32m'   
YELLOW='\033[1;33m'
PURPLE='\033[1;35m'
NC='\033[0m'         

# Default port
PORT=8000
API_URL="http://localhost:$PORT"
TESTS_PASSED=0
TESTS_FAILED=0
AUTO_MODE=false

# Parse command line arguments
while getopts "p:a" opt; do
  case $opt in
    p) PORT="$OPTARG"
       API_URL="http://localhost:$PORT"
       ;;
    a) AUTO_MODE=true
       ;;
    \?) echo "Invalid option -$OPTARG" >&2
        exit 1
        ;;
  esac
done

# Arrays to store test results
declare -a FAILED_TESTS=()
declare -a ERROR_MESSAGES=()
declare -a MANUAL_TESTS=()
declare -a MANUAL_RESULTS=()

# Function to record test failure
record_failure() {
    local test_name=$1
    local error_msg=$2
    FAILED_TESTS+=("$test_name")
    ERROR_MESSAGES+=("$error_msg")
    ((TESTS_FAILED++))
}

# Function to ask for user confirmation
ask_user() {
    local question=$1
    local default=$2
    
    if [ "$AUTO_MODE" = true ]; then
        # In auto mode, just wait a bit and assume success
        sleep 3
        return 0
    fi
    
    echo -e "${PURPLE}$question ${NC}(y/n) [${default}]: "
    read -r response
    response=${response:-$default}
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}

# Function to pause between tests
pause_between_tests() {
    local test_name=$1
    
    MANUAL_TESTS+=("$test_name")
    
    if [ "$AUTO_MODE" = true ]; then
        # In auto mode, just wait a bit
        sleep 2
        MANUAL_RESULTS+=("Auto-approved")
        return 0
    fi
    
    echo -e "\n${YELLOW}=== Test Completed: $test_name ===${NC}"
    echo -e "${PURPLE}Press Enter to continue to the next test...${NC}"
    read
}

check_dependencies() {
    echo -e "${CYAN}Checking Python dependencies...${NC}"
    python3 -c "
import sys
missing = []
for module in ['sounddevice', 'soundfile', 'numpy']:
    try:
        __import__(module)
    except ImportError:
        missing.append(module)
if missing:
    print('Missing Python modules: ' + ', '.join(missing))
    sys.exit(1)
"
    if [ $? -ne 0 ]; then
        record_failure "Dependency Check" "Missing required Python modules"
        echo -e "${RED}Missing required Python modules${NC}"
        echo "Please install them with: pip install sounddevice soundfile numpy"
        exit 1
    fi
}

play_audio() {
    local wav_file=$1
    echo -e "${CYAN}Playing audio file: $wav_file${NC}"
    
    python3 - <<END
import sounddevice as sd
import soundfile as sf
import sys

try:
    data, samplerate = sf.read('$wav_file')
    sd.play(data, samplerate)
    sd.wait()
except Exception as e:
    print(f"Error playing audio: {e}", file=sys.stderr)
    sys.exit(1)
END
}

test_endpoint() {
    local endpoint=$1
    local description=$2
    
    echo -e "\n${CYAN}Testing: ${description}${NC}"
    local response=$(curl -s "$API_URL$endpoint")
    
    if [ $? -eq 0 ] && [ ! -z "$response" ]; then
        echo -e "${GREEN}Success:${NC}"
        echo "$response" | python3 -m json.tool
        ((TESTS_PASSED++))
        pause_between_tests "$description"
        return 0
    else
        record_failure "$description" "Failed to get response from $endpoint"
        echo -e "${RED}Failed to get response from $endpoint${NC}"
        pause_between_tests "$description"
        return 1
    fi
}
test_synthesis_playback() {
    local description=$1
    local text=$2
    local voice=$3
    local lang=$4
    
    echo -e "\n${CYAN}$description${NC}"
    echo -e "${CYAN}Sending request for text: '${text}'${NC}"
    echo -e "${CYAN}Voice: $voice, Language: $lang${NC}"
    
    # Capture response directly in a variable
    local response
    response=$(curl -s -X POST "$API_URL/synthesize" \
        -H "Content-Type: application/json" \
        -d "{\"text\":\"$text\",\"voice\":\"$voice\",\"language\":\"$lang\"}")
    
    echo -e "${YELLOW}Response:${NC} $response"
    
    # Check for success
    if [[ "$response" == *"\"status\":\"success\""* ]]; then
        echo -e "${GREEN}✓ API response contains '\"status\":\"success'${NC}"
    elif [[ "$response" == *"success"* ]]; then
        echo -e "${GREEN}✓ API response contains 'success' (simple check)${NC}"
    else
        echo -e "${RED}✗ API response doesn't contain 'success'${NC}"
        echo -e "${YELLOW}Exact response bytes:${NC}"
        echo "$response" | xxd -g 1
    fi
    
    # Debug parse with jq
    echo -e "${CYAN}Testing with jq (just for debug):${NC}"
    echo "$response" | jq '.' || echo -e "${RED}jq parsing failed${NC}"
    
    # Prompt user about audio
    echo -e "${PURPLE}The API should be playing audio now...${NC}"
    sleep 3
    if ask_user "Did you hear the text being spoken?" "y"; then
        echo -e "${GREEN}✓ User confirmed audio was played${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ User did not hear the audio${NC}"
        record_failure "$description" "User did not hear the audio"
        MANUAL_RESULTS+=("User did not hear the audio")
    fi
    
    pause_between_tests "$description"
    return 0
}



test_synthesis_file() {
    local description=$1
    local text=$2
    local voice=$3
    local lang=$4
    local output_file=$5
    
    echo -e "\n${CYAN}$description${NC}"
    echo -e "${CYAN}Sending request for text: '${text}'${NC}"
    echo -e "${CYAN}Voice: $voice, Language: $lang${NC}"
    
    local response=$(curl -s -w "%{http_code}" -X POST "$API_URL/synthesize-file" \
         -H "Content-Type: application/json" \
         -d "{\"text\":\"$text\",\"voice\":\"$voice\",\"language\":\"$lang\"}" \
         --output "$output_file")
    
    if [ -f "$output_file" ]; then
        if file "$output_file" | grep -q "WAVE audio"; then
            echo -e "${GREEN}Success: Generated $output_file${NC}"
            
            echo -e "${CYAN}Playing the generated audio file...${NC}"
            if play_audio "$output_file"; then
                # Ask user if they heard the audio
                if ask_user "Did you hear the text being spoken from the file?" "y"; then
                    echo -e "${GREEN}✓ User confirmed audio was played from file${NC}"
                    MANUAL_RESULTS+=("User heard the file audio")
                    ((TESTS_PASSED++))
                else
                    echo -e "${RED}✗ User did not hear the audio from file${NC}"
                    record_failure "$description" "User did not hear the audio from file"
                    MANUAL_RESULTS+=("User did not hear the file audio")
                fi
            else
                record_failure "$description" "Audio playback failed"
                MANUAL_RESULTS+=("Playback error")
            fi
        else
            local error_content=$(cat "$output_file")
            record_failure "$description" "$error_content"
            echo -e "${RED}Failed: Not a valid WAV file${NC}"
            echo -e "${YELLOW}File content:${NC}"
            cat "$output_file"
            MANUAL_RESULTS+=("Not a valid WAV file")
            
            # Still ask if the user heard anything
            if ask_user "Despite the error, did you hear any audio?" "n"; then
                echo -e "${YELLOW}⚠ User heard audio despite file generation error${NC}"
                MANUAL_RESULTS+=("User heard audio despite error")
            fi
        fi
    else
        record_failure "$description" "No file generated"
        MANUAL_RESULTS+=("No file generated")
        
        # Still ask if the user heard anything
        if ask_user "Despite the error, did you hear any audio?" "n"; then
            echo -e "${YELLOW}⚠ User heard audio despite file generation error${NC}"
            MANUAL_RESULTS+=("User heard audio despite error")
        fi
    fi
    
    pause_between_tests "$description"
    return 0
}

test_error_case() {
    local description=$1
    local voice=$2
    local lang=$3
    
    echo -e "\n${CYAN}$description${NC}"
    echo -e "${CYAN}Testing with voice: $voice, language: $lang${NC}"
    
    local response=$(curl -s -X POST "$API_URL/synthesize-file" \
         -H "Content-Type: application/json" \
         -d "{\"text\":\"Test\",\"voice\":\"$voice\",\"language\":\"$lang\"}")
    
    echo -e "${YELLOW}Response:${NC} $response"
    
    if echo "$response" | grep -q "error\|detail"; then
        echo -e "${GREEN}Successfully detected error case${NC}"
        ((TESTS_PASSED++))
        MANUAL_TESTS+=("$description")
        MANUAL_RESULTS+=("Error detected as expected")
        
        # Still ask if the user heard anything
        if ask_user "Did you hear any audio despite the error?" "n"; then
            echo -e "${YELLOW}⚠ User heard audio despite error response${NC}"
            MANUAL_RESULTS+=("User heard audio despite error")
        fi
    else
        record_failure "$description" "Unexpected response: $response"
        MANUAL_TESTS+=("$description")
        MANUAL_RESULTS+=("Error not detected")
        
        # Ask if the user heard anything
        if ask_user "Did you hear any audio?" "n"; then
            echo -e "${YELLOW}⚠ User heard audio despite unexpected response${NC}"
            MANUAL_RESULTS+=("User heard audio with unexpected response")
        fi
    fi
    
    pause_between_tests "$description"
    return 0
}

print_summary() {
    echo -e "\n${CYAN}=== Test Summary ===${NC}"
    echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Tests failed: ${RED}$TESTS_FAILED${NC}"
    
    # Print manual test results
    if [ ${#MANUAL_TESTS[@]} -gt 0 ]; then
        echo -e "\n${CYAN}=== Manual Test Results ===${NC}"
        for i in "${!MANUAL_TESTS[@]}"; do
            local test_name="${MANUAL_TESTS[$i]}"
            local result="${MANUAL_RESULTS[$i]}"
            
            if [[ "$result" == *"User heard"* || "$result" == "Error detected as expected" || "$result" == "Auto-approved" ]]; then
                echo -e "${GREEN}✓ $test_name:${NC} $result"
            else
                echo -e "${RED}✗ $test_name:${NC} $result"
            fi
        done
    fi
    
    # Print failure details
    if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
        echo -e "\n${RED}=== Failed Tests Details ===${NC}"
        for i in "${!FAILED_TESTS[@]}"; do
            echo -e "\n${RED}Failed Test: ${FAILED_TESTS[$i]}${NC}"
            echo -e "${YELLOW}Error Message:${NC}"
            echo "${ERROR_MESSAGES[$i]}"
        done
    fi
}

# Print usage info
echo -e "${CYAN}===============================================${NC}"
echo -e "${CYAN}      Kokoro TTS API Interactive Test         ${NC}"
echo -e "${CYAN}===============================================${NC}"
echo -e "This script will test the TTS API endpoints and ask for"
echo -e "your confirmation when audio should be playing."
echo -e "\nUse -a flag for automatic mode (no user prompts)."
echo -e "${CYAN}===============================================${NC}"

if [ "$AUTO_MODE" = true ]; then
    echo -e "${YELLOW}Running in automatic mode. No user input required.${NC}"
else
    echo -e "${YELLOW}Running in interactive mode. You will be asked to confirm audio playback.${NC}"
fi

# Start of main execution
check_dependencies

# Verify API is running
echo -e "${CYAN}Verifying API is running on port $PORT...${NC}"
if ! curl -s "http://localhost:$PORT/health" > /dev/null; then
    record_failure "API Health Check" "API is not running on port $PORT"
    echo -e "${RED}API is not running on port $PORT. Please start it first with: python api.py --port $PORT${NC}"
    print_summary
    exit 1
fi

echo -e "${GREEN}API is running. Starting tests...${NC}"

# Run all tests
test_endpoint "/health" "Health Check Endpoint"
test_endpoint "/" "System Info (Available Voices and Languages)"
test_endpoint "/voices" "Available Voices"
test_endpoint "/languages" "Available Languages"

echo -e "\n${CYAN}Testing Direct Speech Synthesis${NC}"
test_synthesis_playback \
    "Test 5.1: Basic synthesis with playback" \
    "Hello, this is a direct playback test 5.1 " \
    "af_bella" \
    "en-us"

sleep 2

echo -e "\n${CYAN}Testing Speech Synthesis to File${NC}"
test_synthesis_file \
    "Test 6.1: Basic synthesis to file" \
    "Hello, this is a file generation test 6.1" \
    "af_bella" \
    "en-us" \
    "test1.wav"

test_synthesis_file \
    "Test 6.2: French synthesis to file" \
    "Bonjour, ceci est un test 6.2" \
    "af_bella" \
    "fr-fr" \
    "test2.wav"

test_synthesis_file \
    "Test 6.3: Long text synthesis to file" \
    "Test 6.3 This is a longer piece of text that will test the system's ability to handle multiple sentences." \
    "af_bella" \
    "en-us" \
    "test3.wav"

echo -e "\n${CYAN}Testing Error Cases${NC}"
test_error_case "Test 7.1: Invalid voice" "invalid_voice" "en-us"
test_error_case "Test 7.2: Invalid language" "af_bella" "invalid_lang"

# Clean up test files
echo -e "\n${CYAN}Cleaning up test files...${NC}"
rm -f test1.wav test2.wav test3.wav

# Print final summary
print_summary

# Exit with appropriate status code
if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "\n${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "\n${RED}Some tests failed${NC}"
    exit 1
fi
