import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import main



if __name__ == "__main__":
    # Test English audio mode with EPDS
    sys.argv = [
        'main.py',
        #'--audio',  
        # '--openai-api-key', os.environ.get('OPENAI_API_KEY'),
        '--interactive', 'true',
        '--groq_api_key',os.environ.get('GROQ_API_KEY'),

        # '--assistant_model', 'llama-3.3-70b-versatile',
        '--assistant_model', 'openai/gpt-oss-120b',
        '--assistant_provider', 'groq',

        # '--patient_model','llama-3.3-70b-versatile',
        '--patient_model','openai/gpt-oss-120b',
        '--patient_provider','groq'
    ]
    print("Running test...")
    main()