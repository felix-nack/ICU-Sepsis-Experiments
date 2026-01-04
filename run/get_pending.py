"""Helper script to get pending experiment indices"""
import sys
import os

# Suppress stdout temporarily to avoid capturing print statements from utilities
from io import StringIO

sys.path.append(os.getcwd())
from src.utils.run_utils import get_list_pending_experiments
from src.utils.json_handling import get_sorted_dict, get_param_iterable

if __name__ == "__main__":
    json_file = sys.argv[1]
    overwrite = sys.argv[2].lower() == 'true' if len(sys.argv) > 2 else False
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    end = int(sys.argv[4]) if len(sys.argv) > 4 else -1
    
    d = get_sorted_dict(json_file)
    experiments = get_param_iterable(d)
    
    # Suppress print statements from get_list_pending_experiments
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    if not overwrite:
        pending = get_list_pending_experiments(experiments)
    else:
        pending = list(range(len(experiments)))
    
    # Restore stdout
    sys.stdout = old_stdout
    
    # Filter by start/end
    filtered = []
    for idx in pending:
        if end > -1:
            if idx >= start and idx <= end:
                filtered.append(idx)
        else:
            if idx >= start:
                filtered.append(idx)
    
    print(','.join(map(str, filtered)))
