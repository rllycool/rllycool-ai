import os

SOURCE_FOLDER = 'Messages Copy'
OUTPUT_FILE = 'data/all_messages.txt'

def collect():
    count = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for root, _, files in os.walk(SOURCE_FOLDER):
            if 'messages.json' in files:
                path = os.path.join(root, 'messages.json')
                
                with open(path, 'r', encoding='utf-8') as f_in:
                    message_content = '"Contents": "'
                    for line in f_in:
                        if message_content in line:
                            start_idx = line.find(message_content) + len(message_content)
                            end_idx = line.find('", "', start_idx)
                            
                            if end_idx != -1:
                                message = line[start_idx:end_idx]
                                f_out.write(message + '\n')
                                count += 1

    print(f"Finished! {count} messages extracted manually.")

if __name__ == "__main__":
    collect()