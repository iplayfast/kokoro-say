#!/bin/bash

# Colors for output
CYAN='\033[1;36m'
GREEN='\033[1;32m'
RED='\033[1;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

LONG_TEXT="This is a very long piece of text that will take some time to speak. It contains multiple sentences and should take at least 10 seconds to say. Let me tell you about text to speech systems. They convert written text into spoken words. This technology has many applications in accessibility, education, and entertainment. The quality of synthetic speech has improved dramatically over the years."
INTERRUPT_TEXT="This is an interruption!"

wait_key() {
    echo -e "\n${YELLOW}Press Enter to continue...${NC}"
    read
}

play_file() {
    local file=$1
    echo -e "${CYAN}Playing file: $file${NC}"
    python3 -c "
import soundfile as sf
import sounddevice as sd
data, samplerate = sf.read('$file')
sd.play(data, samplerate)
sd.wait()
"
}

run_test() {
    local test_num=$1
    local description=$2
    shift 2
    echo -e "\n${CYAN}Test ${test_num}: ${description}${NC}"
    echo -e "${GREEN}Running: $@${NC}"
    "$@"
}

echo -e "killing any old models"
python say.py --kill

echo -e "${CYAN}Basic Tests${NC}"

run_test "1.1" "Basic English synthesis with af_bella" \
    python say.py --voice af_bella --lang en-us "Test 1.1: This is a basic English synthesis test"
wait_key

run_test "2.1" "Alternative voice test with af_nichole" \
    python say.py --voice af_nicole --lang en-us "Test 2.1: Testing with a different voice"
wait_key

run_test "3.1" "French synthesis with ff_siwis" \
    python say.py --voice ff_siwis --lang fr-fr "Test 3.1: Bonjour, ceci est un test en français"
wait_key

echo -e "\n${CYAN}File Output Tests${NC}"

run_test "4.1" "Basic file output af_bella" \
    python say.py --voice af_bella --lang en-us --output test4_1.wav "Test 4.1: This is a basic file output test"
wait_key
play_file "test4_1.wav"
wait_key

run_test "4.2" "French file output af_bella" \
    python say.py --voice af_bella --lang fr-fr --output test4_2.wav "Test 4.2: Bonjour, ceci est un test de fichier"
wait_key
play_file "test4_2.wav"
wait_key

echo -e "\n${CYAN}Interruption Tests${NC}"

echo -e "${CYAN}Test 5.1: Same voice interruption test${NC}"
echo -e "${YELLOW}Starting long text - will attempt to interrupt in 2 seconds...${NC}"
python say.py --voice af_bella --lang en-us "$LONG_TEXT" 
sleep 2
echo -e "${YELLOW}Attempting interruption with same voice...${NC}"
python say.py --voice af_bella --lang en-us "$INTERRUPT_TEXT"
wait_key

echo -e "${CYAN}Test 5.2: Different voice concurrent test${NC}"
echo -e "${YELLOW}Starting long text with first voice - second voice will start in 2 seconds...${NC}"
python say.py --voice af_bella --lang en-us "$LONG_TEXT" 
sleep 2
echo -e "${YELLOW}Starting second voice...${NC}"
python say.py --voice af_nicole --lang en-us "This is a different voice speaking at the same time"
wait_key

echo -e "\n${CYAN}Long Text File Tests${NC}"

run_test "6.1" "Long text file output" \
    python say.py --voice af_bella --lang en-us --output test6_1.wav "$LONG_TEXT"
wait_key
play_file "test6_1.wav"
wait_key

run_test "6.2" "Multiple voice file outputs" \
    python say.py --voice af_nicole --lang en-us --output test6_2a.wav "First voice speaking" && \
    python say.py --voice af_bella --lang en-us --output test6_2b.wav "Second voice speaking"
wait_key
echo -e "${CYAN}Playing first voice file...${NC}"
play_file "test6_2a.wav"
wait_key
echo -e "${CYAN}Playing second voice file...${NC}"
play_file "test6_2b.wav"
wait_key

echo -e "\n${CYAN}Error Cases${NC}"

run_test "7.1" "Invalid voice" \
    python say.py --voice invalid_voice --lang en-us "Test 7.1: This should fail"
wait_key

run_test "7.2" "Invalid language" \
    python say.py --voice af_bella --lang invalid_lang "Test 7.2: This should fail"
wait_key

# Clean up files
rm -f test*.wav

echo -e "\n${GREEN}Tests complete!${NC}"
