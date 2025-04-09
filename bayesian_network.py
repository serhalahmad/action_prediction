from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1. Define the model structure
model = DiscreteBayesianNetwork([
    ('GazeDirection', 'TargetObject'),
    ('GripShape', 'TargetObject'),
    ('HandProximity', 'TargetObject'),
    ('Hand3DCloseness', 'TargetObject')
])

# 2. Define state names for all nodes
state_names = {
    'GazeDirection': ['towards_small', 'towards_large'],
    'GripShape': ['small', 'large'],
    'HandProximity': ['near_small', 'near_large'],
    'Hand3DCloseness': ['close', 'far'],
    'TargetObject': ['small', 'large']
}

# 3. Define CPDs for input nodes (uniform for now)
cpd_gaze = TabularCPD('GazeDirection', 2, [[0.5], [0.5]], state_names=state_names)
cpd_grip = TabularCPD('GripShape', 2, [[0.5], [0.5]], state_names=state_names)
cpd_proximity = TabularCPD('HandProximity', 2, [[0.5], [0.5]], state_names=state_names)
cpd_closeness = TabularCPD('Hand3DCloseness', 2, [[0.5], [0.5]], state_names=state_names)  # Now only close and far

# 4. Define full CPD for TargetObject (manually filled)
cpd_values = [
    [0.99, 0.99, 0.3, 0.95, 0.85, 0.95, 0.05, 0.95, 0.95, 0.05, 0.15, 0.05, 0.95, 0.05, 0.01, 0.01],  # Probabilities for "small"
    [0.01, 0.01, 0.7, 0.05, 0.15, 0.05, 0.95, 0.05, 0.05, 0.95, 0.85, 0.95, 0.05, 0.95, 0.99, 0.99]   # Probabilities for "large"
]

# Create the full CPD for TargetObject
cpd_target = TabularCPD(
    variable='TargetObject', variable_card=2,
    values=cpd_values,
    evidence=['GazeDirection', 'GripShape', 'HandProximity', 'Hand3DCloseness'],
    evidence_card=[2, 2, 2, 2],
    state_names=state_names
)

# 5. Add all CPDs to the model
model.add_cpds(cpd_gaze, cpd_grip, cpd_proximity, cpd_closeness, cpd_target)

# 6. Validate the model
assert model.check_model()

# 7. Run inference
infer = VariableElimination(model)

# Example query to predict the TargetObject given evidence
query = infer.query(
    variables=['TargetObject'],
    evidence={
        'GazeDirection': 'towards_small',
        'GripShape': 'large',
        'HandProximity': 'near_large',
        'Hand3DCloseness': 'far'
    }
)

# Print the result
print(query)
print(query.values[0], query.values[1])
