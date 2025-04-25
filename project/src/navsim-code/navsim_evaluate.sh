export PYTHONPATH=/navsim_workspace
# Navtest dataset
python navsim/navsim/planning/script/run_pdm_score_one_stage.py \
    train_test_split=navtest \
    agent=ijepa_agent \
    agent.mlp_weights_path="/navsim_workspace/code/checkpoints/planning_head_20250423_184215_loss0_3079.pth" \
    experiment_name=ijepa_agent_navtest \
    worker=ray_distributed \
    output_dir="code/outputs/ijepa_agent_navtest_results"
    
    
# Test dataset
python navsim/navsim/planning/script/run_pdm_score_one_stage.py \
    train_test_split=test \
    agent=ijepa_agent \
    agent.mlp_weights_path="/navsim_workspace/code/checkpoints/planning_head_20250423_184215_loss0_3079.pth" \
    traffic_agents=non_reactive \
    experiment_name=ijepa_agent_test_nonreactive \
    output_dir="code/outputs/ijepa_agent_navtest_nonreactive_results" \
    worker=ray_distributed

    # Assuming running from /navsim_workspace/
python navsim/navsim/planning/script/run_metric_caching.py \
    train_test_split=test \

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
train_test_split=test \
metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache 

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    experiment_name=training_ijepa_planning_agent \
    trainer.params.max_epochs=50 \
    train_test_split=navtrain

# Assuming NAVSIM_DEVKIT_ROOT is set correctly
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    experiment_name=trainval_ijepa_planning_agent \
    trainer.params.max_epochs=30 \
    train_test_split=trainval \
    worker=ray_distributed
    # You might not need the explicit agent_override=... if you changed the default in the main config
    # If you kept the default agent: ego_status_mlp_agent and want to override from command line:
    # agent_override=ijepa_planning_agent # This syntax uses the override group if defined


TRAIN_TEST_SPLIT=test
MLP_WEIGHTS="/navsim_workspace/code/checkpoints/planning_head_20250423_184215_loss0_3079.pth"
EXPERIMENT_NAME=ijepa_planning_agent_eval_one_stage

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=ijepa_planning_agent \
    agent.mlp_weights_path=$MLP_WEIGHTS \
    experiment_name=$EXPERIMENT_NAME \
    # --------------------------
    # --- CHANGE THESE LINES ---
    # traffic_agents_policy=non_reactive # Keep this if needed
    # Point this to the .pth file containing your trained MLP state_dict

# Define cache path variable (good practice)
CACHE_DIR= # Or adjust path as needed

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
    agent=ijepa_planning_agent \
    train_test_split=trainval \
    cache_path=$NAVSIM_EXP_ROOT/training_cache \
    force_cache_computation=True \
    experiment_name=cache_ijepa_agent_trainval # Optional, for clarity

# --- Configuration ---
# Define paths (use absolute paths or ensure relative paths are correct from execution directory)
LOG_PATH="dataset/trainval_navsim_logs/trainval"
BLOB_PATH="dataset/trainval_sensor_blobs/trainval"
CACHE_PATH="${NAVSIM_EXP_ROOT}/cache_ijepa_agent_trainval" # Make sure NAVSIM_EXP_ROOT is set

# --- Command ---
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    experiment_name=train_ijepa_agent_trainval \
    agent=ijepa_planning_agent \
    train_test_split=trainval \
    trainer.params.max_epochs=30 \
    navsim_log_path=$LOG_PATH \
    original_sensor_path=$BLOB_PATH \
    cache_path="$NAVSIM_EXP_ROOT/training_cache" \
    force_cache_computation=false # Set true only if regenerating cache
    worker=ray_distributed \
    # +trainer.strategy=auto # If you don't need DDP for local run

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
    agent=ijepa_planning_agent \
    train_test_split=trainval \
    cache_path=$NAVSIM_EXP_ROOT/training_cache \
    force_cache_computation=True \
    experiment_name=cache_ijepa_agent_trainval \
    worker=ray_distributed \
    worker.threads_per_node=4 # <--- Limit Ray threads (equivalent to max_workers)