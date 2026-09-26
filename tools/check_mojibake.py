import os
import sys

# Common double-encoded UTF-8 (mojibake) sequences
# UTF-8 decoded as CP1252, then re-encoded as UTF-8
MOJIBAKE_PATTERNS = [
    b'\xc3\xa2\xe2\x80\x94', # â€” (—)
    b'\xc3\xa2\xe2\x80\xa0', # â† (→)
    b'\xc3\xa2\xe2\x80\x9c', # â€œ (“)
    b'\xc3\xa2\xe2\x80\x9d', # â€  (”)
    b'\xc3\xa2\xe2\x80\x9e', # â€ž („)
    b'\xc3\xa2\xe2\x80\xa6', # â€¦ (…)
    b'\xc3\xa2\xe2\x80\xb0', # â€° (‰)
    b'\xc3\xa2\xe2\x80\xb9', # â€¹ (‹)
    b'\xc3\xa2\xe2\x80\xba', # â€º (›)
    b'\xc3\xa2\xe2\x80\x98', # â€˜ (‘)
    b'\xc3\xa2\xe2\x80\x99', # â€™ (’)
    b'\xc3\xa2\xe2\x80\x93', # â€“ (–)
    b'\xc3\xa2\xe2\x84\xa2', # â„¢ (™)
    b'\xc3\xa2\xe2\x80\xa2', # â€¢ (•)
    b'\xc3\xa2\x94\x80',     # â”€ (─)
    b'\xc3\xa2\x95\x90',     # â•  (═)
    b'\xc3\x83\xc2\x97',     # Ã— (×)
    b'\xc3\x8e\xc2\x94',     # Î” (Δ)
]

def check_file(filepath):
    with open(filepath, 'rb') as f:
        content = f.read()
    
    found = []
    for pattern in MOJIBAKE_PATTERNS:
        if pattern in content:
            found.append(pattern)
            
    return found

def main(root_dir):
    print(f"Scanning {root_dir} for mojibake...")
    issues = 0
    for root, dirs, files in os.walk(root_dir):
        if '.git' in dirs:
            dirs.remove('.git')
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')
            
        for file in files:
            if file.endswith(('.py', '.js', '.md')):
                path = os.path.join(root, file)
                mojibake = check_file(path)
                if mojibake:
                    print(f"[FAIL] {path}: Found {len(mojibake)} patterns")
                    issues += 1
                    
    if issues == 0:
        print("Success: No mojibake found.")
        sys.exit(0)
    else:
        print(f"Failed: Found mojibake in {issues} files.")
        sys.exit(1)

if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main(root)
