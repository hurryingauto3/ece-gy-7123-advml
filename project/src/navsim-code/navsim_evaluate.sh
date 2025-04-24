# Run this script from the root of the repository
# /navsim_workspace

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
    +scenario_builder=pdm_test \
    +train_test_split.data_path="dataset/test_navsim_logs/test" \
    train_test_split=test \