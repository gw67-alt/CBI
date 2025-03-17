
from tqdm import tqdm
import threading
from queue import Queue
from typing import Dict, Set, List, Tuple
import os
from collections import deque, defaultdict
import time
import sys
import serial
import serial.tools.list_ports
import re

KB_limit = -1
BUFFER_SIZE = 999999

# Serial connection parameters
BAUD_RATE = 9600
SERIAL_TIMEOUT = 0.1  # Non-blocking read timeout
THRESHOLD = 500  # Default threshold for serial signal
categories = ["what", "how"]
translation_dict = {
    "what": "descriptions.txt",  # nouns (can be subjects or objects)
    "how": "actions.txt",       # adverbs
    "do": "verbs.txt",         # verbs
    "describe": "picturable.txt", # articles/determiners
    "grade": "adj.txt",        # adjectives
    "form": "prep.txt"         # prepositions
}
list_of_words = [] 

class SerialMonitor:
    def __init__(self, port=None, baud_rate=BAUD_RATE, timeout=SERIAL_TIMEOUT, threshold=THRESHOLD):
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.threshold = threshold
        self.serial_conn = None
        self.stop_event = threading.Event()
        self.monitor_thread = None
        self.threshold_exceeded = threading.Event()
    
    def list_ports(self):
        """List available serial ports."""
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]
    
    def connect(self, port=None):
        """Connect to a serial port."""
        if port:
            self.port = port
        
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            print(f"Connected to {self.port} at {self.baud_rate} baud")
            return True
        except Exception as e:
            print(f"Error connecting to serial port: {e}")
            self.serial_conn = None
            return False
    
    def disconnect(self):
        """Disconnect from serial port."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print(f"Disconnected from {self.port}")
    
    def start_monitoring(self):
        """Start monitoring serial data in a separate thread."""
        if not self.serial_conn or not self.serial_conn.is_open:
            print("Cannot start monitoring: Serial connection not established")
            return False
        
        self.stop_event.clear()
        self.threshold_exceeded.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_serial_data)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"Started monitoring serial data")
        return True
    
    def stop_monitoring(self):
        """Stop the serial monitoring thread."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.stop_event.set()
            self.monitor_thread.join(timeout=1.0)
            print("Stopped serial monitoring")
    
    def set_threshold(self, threshold):
        """Set a new threshold value."""
        self.threshold = threshold
        print(f"Threshold set to {threshold}")
    
    def is_threshold_exceeded(self):
        """Check if threshold has been exceeded."""
        return self.threshold_exceeded.is_set()
    
    def reset_threshold_flag(self):
        """Reset the threshold exceeded flag."""
        self.threshold_exceeded.clear()
    
    def _monitor_serial_data(self):
        """Thread function to monitor serial data."""
        while not self.stop_event.is_set():
            if self.serial_conn and self.serial_conn.is_open:
                try:
                    if self.serial_conn.in_waiting > 0:
                        # Read line and convert to integer
                        line = self.serial_conn.readline().decode('utf-8').strip()
                        try:
                            value = int(line)
                            # Check if value exceeds threshold
                            if value > 0:
                                self.threshold_exceeded.set()
                        except ValueError:
                            # Ignore non-integer values
                            pass
                except Exception as e:
                    print(f"Error reading serial data: {e}")
                    time.sleep(0.1)
            time.sleep(0.01)  # Small delay to prevent CPU hogging

class SVOPattern:
    def __init__(self):
        self.subjects = defaultdict(set)       # subject -> verb
        self.verbs = defaultdict(set)          # verb -> object
        self.objects = defaultdict(set)        # object -> subject
        self.subject_object = defaultdict(set) # subject -> object

    def add_pattern(self, subject: str, verb: str, obj: str):
        self.subjects[subject].add(verb)
        self.verbs[verb].add(obj)
        self.objects[obj].add(subject)
        self.subject_object[subject].add(obj)

    def get_verbs_for_subject(self, subject: str) -> Set[str]:
        return self.subjects[subject]

    def get_objects_for_verb(self, verb: str) -> Set[str]:
        return self.verbs[verb]

    def get_subjects_for_object(self, obj: str) -> Set[str]:
        return self.objects[obj]
        
    def save_to_file(self, filename: str):
        with open(filename, "w", encoding="utf-8") as f:
            for subject in self.subjects:
                verbs = self.subjects[subject]
                for verb in verbs:
                    objects = self.verbs[verb]
                    for obj in objects:
                        f.write(f"{subject} {verb} {obj}\n")
                        
    @classmethod
    def load_from_file(cls, filename: str) -> 'SVOPattern':
        pattern = cls()
        try:
            with open(filename, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        subject, verb, obj = parts
                        pattern.add_pattern(subject, verb, obj)
            return pattern
        except FileNotFoundError:
            print(f"File {filename} not found.")
            return None
        except Exception as e:
            print(f"Error loading patterns: {e}")
            return None

class VocabularyCache:
    def __init__(self, translation_dict: Dict[str, str]):
        self.vocab_cache: Dict[str, Set[str]] = {}
        self._load_vocabularies(translation_dict)
    
    def _load_vocabularies(self, translation_dict: Dict[str, str]) -> None:
        for category, filename in translation_dict.items():
            with open(filename, 'r', encoding='utf-8') as f:
                self.vocab_cache[category] = {line.strip() for line in f.readlines()}
    
    def get_vocabulary(self, category: str) -> Set[str]:
        return self.vocab_cache.get(category, set())
    
    def is_word_in_category(self, word: str, category: str) -> bool:
        """Check if a word belongs to a specific category."""
        return word in self.vocab_cache.get(category, set())
    
    def find_word_category(self, word: str) -> str:
        """Find which category a word belongs to."""
        for category, words in self.vocab_cache.items():
            if word in words:
                return category
        return None

def process_sentence(sentence: str, vocab_cache: VocabularyCache, svo_patterns: SVOPattern = None) -> str:
    words = sentence.split()
    temp = "["
    
    # First pass: categorize words and track positions
    word_categories = {}
    for i, word in enumerate(words):
        for category, vocab in vocab_cache.vocab_cache.items():
            if word in vocab:
                word_categories[i] = (word, category)
                temp += f":{category}>{word}"
    
    # Second pass: identify SVO patterns
    if svo_patterns is not None:
        for i in range(len(words)-2):
            if i in word_categories and i+1 in word_categories and i+2 in word_categories:
                word1, cat1 = word_categories[i]
                word2, cat2 = word_categories[i+1]
                word3, cat3 = word_categories[i+2]
                
                # Check for SVO pattern
                if cat1 == "what" and cat2 == "do" and cat3 == "what":
                    svo_patterns.add_pattern(word1, word2, word3)
    
    temp += ":]\n"
    return temp if len(temp) > 3 else ""

import random

def generate_svo_sentence(svo_patterns: SVOPattern, vocab_cache: VocabularyCache, randomize: bool = False) -> str:
    if randomize:
        # Get all subjects that have associated verbs
        valid_subjects = [subj for subj in svo_patterns.subjects.keys() if svo_patterns.subjects[subj]]
        if not valid_subjects:
            return None
            
        # Randomly select a subject
        subject = random.choice(valid_subjects)
        
        # Get verbs associated with this subject and randomly select one
        possible_verbs = list(svo_patterns.get_verbs_for_subject(subject))
        verb = random.choice(possible_verbs)
        
        # Get objects associated with this verb and randomly select one
        possible_objects = list(svo_patterns.get_objects_for_verb(verb))
        obj = random.choice(possible_objects)
        
        return f"{subject} {verb} {obj}."
    else:
        # Pattern-based SVO generation
        for subject in svo_patterns.subjects:
            verbs_for_subject = svo_patterns.get_verbs_for_subject(subject)
            if verbs_for_subject:
                for verb in verbs_for_subject:
                    objects = svo_patterns.get_objects_for_verb(verb)
                    if objects:
                        obj = next(iter(objects))
                        return f"{subject} {verb} {obj}."
    
    return None

def print_word_by_word(sentence: str, delay: float = 1.0, serial_monitor: SerialMonitor = None) -> bool:
    """
    Print a sentence one word at a time with a delay between words.
    Returns True if completed, False if interrupted by serial signal.
    """
    if not sentence:
        return True
   
    words = sentence.split()
    for i, word in enumerate(words):
        # Check serial monitor if provided
        list_of_words.append(word)
        # Print word without newline
        sys.stdout.write(word)
        sys.stdout.flush()
        if serial_monitor and serial_monitor.is_threshold_exceeded():
            serial_monitor.reset_threshold_flag()
            print()
    

            return list_of_words,False
            
        
        
        # Add space after word (except for last word and punctuation)
        if i < len(words) - 1 and not words[i+1] in ['.', ',', '!', '?', ';', ':']:
            sys.stdout.write(' ')
            sys.stdout.flush()
            
        # Delay between words
        time.sleep(delay)

    # Print newline at the end
    return list_of_words,True

class ResultBuffer:
    def __init__(self, output_file: str, buffer_size: int = BUFFER_SIZE):
        self.output_file = output_file
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.lock = threading.Lock()
        self.flush_count = 0
    
    def add_result(self, result: str) -> None:
        with self.lock:
            self.buffer.append(result)
            if len(self.buffer) >= self.buffer_size:
                self.flush_buffer()
    
    def flush_buffer(self) -> None:
        if not self.buffer:
            return
            
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                while self.buffer:
                    f.write(self.buffer.popleft())
            self.flush_count += 1
        except Exception as e:
            print(f"Error writing to file: {e}")
            self.buffer.extendleft(reversed(list(self.buffer)))
    
    def final_flush(self) -> None:
        with self.lock:
            self.flush_buffer()

def worker(sentence_queue: Queue, result_buffer: ResultBuffer, 
          vocab_cache: VocabularyCache, svo_patterns: SVOPattern,
          pbar: tqdm) -> None:
    while True:
        try:
            sentence = sentence_queue.get_nowait()
        except Queue.Empty:
            break
            
        if sentence is None:
            break
            
        result = process_sentence(sentence, vocab_cache, svo_patterns)
        if result:
            result_buffer.add_result(result)
        pbar.update(1)
        sentence_queue.task_done()

def build_memory_multithreaded(filename: str, num_threads: int = None) -> SVOPattern:
    if num_threads is None:
        num_threads = os.cpu_count() or 4
    
    print(f"\nBuilding memory using {num_threads} threads...")
    
    sentence_queue = Queue()
    result_buffer = ResultBuffer("memory.txt")
    svo_patterns = SVOPattern()
    vocab_cache = VocabularyCache(translation_dict)
    
    with open(filename, 'r', encoding='latin-1') as f:
        text = ' '.join(f.read().split()[:KB_limit])
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    
    for sentence in sentences:
        sentence_queue.put(sentence)
    
    for _ in range(num_threads):
        sentence_queue.put(None)
    
    pbar = tqdm(total=len(sentences), desc="Processing Sentences", unit="sentence")
    
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(
            target=worker,
            args=(sentence_queue, result_buffer, vocab_cache, svo_patterns, pbar)
        )
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
    
    result_buffer.final_flush()
    pbar.close()
    
    print(f"\nMemory building complete. Buffer flushed {result_buffer.flush_count} times.")
    return svo_patterns

def print_query_results_word_by_word(results: set, delay: float = 1.0, serial_monitor: SerialMonitor = None) -> tuple:
    """
    Print query results one word at a time.
    Returns (completed_flag, last_word) - where completed_flag is True if printing completed
    and last_word is the last word printed (or None if there were no results).
    """
    if not results:
        print("[ No results found ]")
        return True, None
        
    words = list(results)
    results_str = "[ " + ' '.join(words) + " ]"
    list_of_words, completed = print_word_by_word(results_str, delay, serial_monitor)
    
    # If we have results, return the last actual word
    if words:

        return completed, list_of_words[-1]
    return completed, None

def setup_serial_connection():
    """Setup serial connection with user input."""
    serial_monitor = SerialMonitor()
    available_ports = serial_monitor.list_ports()
    
    if not available_ports:
        print("No serial ports found. Serial monitoring will be disabled.")
        return None
    
    print("\nAvailable serial ports:")
    for i, port in enumerate(available_ports):
        print(f"{i+1}. {port}")
    
    try:
        choice = input(f"Select port (1-{len(available_ports)}, or Enter to skip): ").strip()
        if not choice:
            print("Serial monitoring disabled.")
            return None
            
        port_idx = int(choice) - 1
        if 0 <= port_idx < len(available_ports):
            selected_port = available_ports[port_idx]
            
            if serial_monitor.connect(selected_port):
                if serial_monitor.start_monitoring():
                    return serial_monitor
    except ValueError:
        pass
    
    print("Failed to set up serial monitoring. It will be disabled.")
    return None

def auto_chain_queries(vocab_cache, word_delay, serial_monitor=None, num_iterations=10000, initial_category="what", initial_word=None):
    
    """
    Run a series of queries that automatically chain from one to the next.
    Each query uses the last result from the previous query, alternating categories.
    """
    # Pattern of categories to cycle through

    current_category_idx = categories.index(initial_category) if initial_category in categories else 0
    
    # If no initial word is provided, get a random one from the current category
    current_word = initial_word
    if not current_word:
        category_vocab = vocab_cache.get_vocabulary(categories[current_category_idx])
        if category_vocab:
            svo_patterns = SVOPattern.load_from_file("SVO.txt")
            rand_words = generate_svo_sentence(svo_patterns, vocab_cache, randomize=True).split()
            current_word = random.choice(list(rand_words))
            current_word = re.sub(r'[^\w\s]', '', current_word)
            with open("output.txt", "a") as file:
                # Write the string to the file
                file.write(current_word + " ")
        else:
            print(f"No words found in category '{categories[current_category_idx]}'")
            return
    
    print(f"\nStarting auto-chain with: {categories[current_category_idx]} {current_word}")
    print(f"Will run for {num_iterations} iterations in pattern: {' → '.join(categories)} → ...")
    time.sleep(2)  # Give user time to read explanation
    
    try:
        with open("memory.txt", "r", encoding="utf-8") as f:
            data = f.readlines()
    except FileNotFoundError:
        print("Error: memory.txt not found. Please build memory first (option 1).")
        return
    
    for i in range(num_iterations):
        if not current_word:
            print("\nNo word to continue with. Stopping auto-chain.")
            break
            
        # Get current category
        current_category = categories[current_category_idx]
        print()
        print("Category:", current_category)
        next_category_idx = (current_category_idx + 1) % len(categories)
        next_category = categories[next_category_idx]
        
        # Process query
        out = set()
        interrupted = False
        
        for sub_text in data:
            if serial_monitor and serial_monitor.is_threshold_exceeded():
                serial_monitor.reset_threshold_flag()
                interrupted = True
                break
                
            lingual = sub_text.split(":")
            words_in_entry = {}
            
            for group in lingual:
                parts = group.split(">")
                if len(parts) > 1:
                    element0 = parts[0].strip()
                    element1 = parts[1].strip()

                    if element0 and element1:
                        if element0 not in words_in_entry:
                            words_in_entry[element0] = set()
                        words_in_entry[element0].add(element1)

            relationship_mappings = [
                ("what", "do"),
                ("how", "do"),
                ("describe", "what"),
                ("grade", "what"),
                ("what", "how"),
                ("describe", "grade"),
                ("how", "what"),
                ("form", "what"),
                ("form", "describe"),
                ("form", "grade"),
                ("form", "how")
            ]

            for source, target in relationship_mappings:
                if current_word in words_in_entry.get(source, set()):
                    # Only collect words from the next category we want
                    if target == next_category:
                        out.update(words_in_entry.get(target, set()))
                    # If we can't find words in the next desired category,
                    # just gather all relevant words for fallback
                    else:
                        out.update(words_in_entry.get(target, set()))
        
        # Filter results to only include words from the next category
        filtered_results = set()
        for word in out:
            if vocab_cache.is_word_in_category(word, next_category):
                filtered_results.add(word)
                
        # If we don't have any words in our target category, use any results
        if not filtered_results:
            filtered_results = out
        
        # Print results and get the last word
        completed, last_word = print_query_results_word_by_word(filtered_results, word_delay, serial_monitor)
        # Move to next category and word
        current_category_idx = next_category_idx
        current_word = last_word
        with open("output.txt", "a") as file:
            # Write the string to the file
            file.write(" -> " + str(last_word) + "\n")
        # If we got no results, try to pick a random word from the next category
        if not current_word:
            category_vocab = vocab_cache.get_vocabulary(categories[current_category_idx])
            if category_vocab:
                svo_patterns = SVOPattern.load_from_file("SVO.txt")
                rand_words = generate_svo_sentence(svo_patterns, vocab_cache, randomize=True).split()
                current_word = random.choice(list(rand_words))
                current_word = re.sub(r'[^\w\s]', '', current_word)
                with open("output.txt", "a") as file:
                    # Write the string to the file
                    file.write(current_word + " ")
                print(f"\nNo results found. Randomly selected new word: {current_word}")
        
        # Small pause between iterations
        time.sleep(1)
    
    print("\nAuto-chain complete.")

def main():
    print(translation_dict)
    svo_patterns = None
    vocab_cache = None
    serial_monitor = None
    
    # Default delay in seconds for word-by-word printing
    word_delay = 0.3
    
    try:
        # Try to load vocabulary at startup
        vocab_cache = VocabularyCache(translation_dict)
    except Exception as e:
        print(f"Warning: Could not load vocabulary files: {e}")
        print("Make sure all required vocabulary files exist in the current directory.")
    
    while True:
        print("\nOptions:")
        print("1. Build memory")
        print("2. Execute queries")
        print("3. Set word display delay (currently {:.1f} seconds)".format(word_delay))
        print("4. Configure serial monitoring" + 
              (f" (active on {serial_monitor.port})" 
               if serial_monitor else " (inactive)"))
        print("5. Auto-chain queries (what-how pattern)")
        print("6. Exit")

        choice = input("\nEnter your choice (1-6): ").strip()
        
        # Ensure vocabulary is loaded
        if not vocab_cache:
            try:
                vocab_cache = VocabularyCache(translation_dict)
            except Exception as e:
                print(f"Error loading vocabulary: {e}")
                continue

        if choice == "1":
            filename = input("Enter training file path: ")
            num_threads = input("Enter number of threads (press Enter for auto): ").strip()
            num_threads = int(num_threads) if num_threads else None
            svo_patterns = build_memory_multithreaded(filename, num_threads)
            svo_patterns.save_to_file("SVO.txt")

        elif choice == "2":
            while True:
                query = input("Enter command (or 'back' to return to main menu): ").split()
                if query[0].lower() == 'back':
                    break
                    
                if len(query) < 2:
                    print("Please enter a category (e.g., 'what') and a word (e.g., 'helped').")
                    continue

                category_to_search = query[0]
                search_words = query

                if category_to_search not in translation_dict:
                    print(f"Invalid category: {category_to_search}")
                    continue

                out = set()
                try:
                    with open("memory.txt", "r", encoding="utf-8") as f:
                        data = f.readlines()

                    with tqdm(total=len(data), desc="Processing data", unit="segment") as pbar:
                        for sub_text in data:
                            pbar.update(1)

                            lingual = sub_text.split(":")
                            words_in_entry = {}
                            for search_word in search_words:
                                for group in lingual:
                                    parts = group.split(">")
                                    if len(parts) > 1:
                                        element0 = parts[0].strip()
                                        element1 = parts[1].strip()

                                        if element0 and element1:
                                            if element0 not in words_in_entry:
                                                words_in_entry[element0] = set()
                                            words_in_entry[element0].add(element1)

                            relationship_mappings = [
                                ("what", "do"),
                                ("how", "do"),
                                ("describe", "what"),
                                ("grade", "what"),
                                ("what", "how"),
                                ("describe", "grade"),
                                ("how", "what"),
                                ("form", "what"),
                                ("form", "describe"),
                                ("form", "grade"),
                                ("form", "how")
                            ]

                            for source, target in relationship_mappings:
                                if search_word in words_in_entry.get(source, set()):
                                    out.update(words_in_entry.get(target, set()))
                except FileNotFoundError:
                    print("Error: memory.txt not found. Please build memory first (option 1).")
                    continue

                # Print results word by word
                print_query_results_word_by_word(out, word_delay, serial_monitor)
               
                
        
        elif choice == "3":
            try:
                new_delay = float(input("Enter new delay in seconds between words (e.g., 0.5): "))
                if new_delay < 0:
                    print("Delay must be a positive number. Setting to 0.1 seconds.")
                    word_delay = 0.1
                else:
                    word_delay = new_delay
                print(f"Word display delay set to {word_delay:.1f} seconds.")
            except ValueError:
                print("Invalid input. Delay remains at {:.1f} seconds.".format(word_delay))
        
        elif choice == "4":
            if serial_monitor:
                print("\nSerial monitoring is currently active.")
                sub_choice = input("1. Reconfigure serial connection\n2. Change threshold\n3. Disable monitoring\nEnter choice: ").strip()
                
                if sub_choice == "1":
                    serial_monitor.stop_monitoring()
                    serial_monitor.disconnect()
                    serial_monitor = setup_serial_connection()
                
                elif sub_choice == "2":
                    try:
                        new_threshold = int(input(f"Current threshold is {serial_monitor.threshold}. Enter new threshold: "))
                        serial_monitor.set_threshold(new_threshold)
                    except ValueError:
                        print("Invalid input. Threshold remains unchanged.")
                
                elif sub_choice == "3":
                    serial_monitor.stop_monitoring()
                    serial_monitor.disconnect()
                    serial_monitor = None
                    print("Serial monitoring disabled.")
            else:
                # Set up new serial connection
                serial_monitor = setup_serial_connection()
        
        elif choice == "5":
            # Auto-chain queries in what-how pattern
            if not os.path.exists("memory.txt"):
                print("Error: memory.txt not found. Please build memory first (option 1).")
                continue
                
            print("\nAuto-Chain Query Configuration")
            try:
                iterations = int(input("Enter number of iterations to run (default 10000): ") or "10000")
            except ValueError:
                print("Invalid input. Using default value of 10 iterations.")
                iterations = 10000
                
            # Get starting category
            start_category = input("Enter starting category ('what' or 'how', default 'what'): ").lower() or "what"
            if start_category not in ["what", "how"]:
                print(f"Invalid category '{start_category}'. Using 'what' instead.")
                start_category = "what"
                
            # Get optional starting word
            start_word = input(f"Enter starting word (or leave empty for random {start_category} word): ").strip()
            if start_word and not vocab_cache.is_word_in_category(start_word, start_category):
                print(f"Warning: '{start_word}' is not in category '{start_category}'. Will use it anyway.")
                
            # Start auto-chaining
            auto_chain_queries(
                vocab_cache=vocab_cache,
                word_delay=word_delay,
                serial_monitor=serial_monitor,
                num_iterations=iterations,
                initial_category=start_category,
                initial_word=start_word
            )
           
        elif choice == "6":
            if serial_monitor:
                serial_monitor.stop_monitoring()
                serial_monitor.disconnect()
            print("Exiting program...")
            break

if __name__ == "__main__":
    main()
