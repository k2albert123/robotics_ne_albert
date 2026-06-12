import runpy
import sys
import io

# Provide 'cpu' as the interactive selection for provider prompts
sys.stdin = io.StringIO('cpu\n')
# Run the recognizer script as __main__ so argparse works
runpy.run_path('addons/mqtt_servo_tracking/recognize_mqtt.py', run_name='__main__')
