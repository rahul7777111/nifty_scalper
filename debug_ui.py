
import sys
import traceback
# Ensure we are in the correct path
sys.path.append(".")

try:
    print("Importing ScalperUI...")
    from src.ui import ScalperUI
    print("Instantiating ScalperUI...")
    app = ScalperUI()
    print("Entering mainloop...")
    app.mainloop()
except:
    # Attempt to restore streams if they were hijacked
    try:
        if hasattr(app, '_orig_stdout'):
            sys.stdout = app._orig_stdout
        else:
            sys.stdout = sys.__stdout__
            
        if hasattr(app, '_orig_stderr'):
            sys.stderr = app._orig_stderr
        else:
            sys.stderr = sys.__stderr__
    except:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        
    print("\n--- CAUGHT EXCEPTION ---")
    traceback.print_exc()
