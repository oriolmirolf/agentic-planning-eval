import os
import subprocess
from dotenv import load_dotenv

# 1. Load .env explicitly (just like the backend should)
load_dotenv()

def check_val():
    print("--- VAL CONFIGURATION CHECK ---")
    
    # 2. Get path from env
    val_path = os.getenv("VAL_PATH")
    print(f"1. VAL_PATH from env: '{val_path}'")

    if not val_path:
        print("❌ ERROR: VAL_PATH is missing in environment.")
        return

    # 3. Check if file exists
    if not os.path.exists(val_path):
        print(f"❌ ERROR: File not found at {val_path}")
        return
    else:
        print(f"✅ File found at {val_path}")

    # 4. Try running it (simple help command)
    print("\n2. Attempting to run VAL...")
    try:
        # VAL usually just prints help if run with no args, or we can use --help if supported
        # We'll just run it bare to see if it executes.
        result = subprocess.run(
            [val_path], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        print("✅ VAL execution successful!")
        print("--- STDOUT ---")
        print(result.stdout[:200]) # First 200 chars
        print("--- STDERR ---")
        print(result.stderr[:200])
        
    except FileNotFoundError:
        print("❌ ERROR: Subprocess failed. The path exists, but it might not be executable.")
    except subprocess.TimeoutExpired:
        print("❌ ERROR: Timed out. VAL is hanging.")
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    check_val()