import argparse

import numpy as np
import psyneulink as pnl

import logging

# This implements the model by Cohen, J. D., & Huston, T. A. (1994). Progress in the use of interactive
# models for understanding attention and performance. In C. Umilta & M. Moscovitch(Eds.),
# AttentionandperformanceXV(pp.453-456). Cam- bridge, MA: MIT Press.
# The model aims to capute top-down effects of selective attention and the bottom-up effects of attentional capture.


parser = argparse.ArgumentParser()
parser.add_argument('--no-plot', action='store_false', help='Disable plotting', dest='enable_plot')
parser.add_argument('--threshold', type=float, help='Termination threshold for response output (default: %(default)f)', default=0.55)
parser.add_argument('--settle-trials', type=int, help='Number of trials for composition to initialize and settle (default: %(default)d)', default=50)
args = parser.parse_args()

# Define Variables ----------------------------------------------------------------------------------------------------
rate = 0.1          # modified from the original code from 0.01 to 0.1
inhibition = -2.0   # lateral inhibition
bias = 4.0          # bias is positive since Logistic equation has - sing already implemented
threshold = args.threshold    # modified from thr original code from 0.6 to 0.55 because incongruent condition won't reach 0.6
settle_trials = args.settle_trials  # cycles until model settles

# Create mechanisms ---------------------------------------------------------------------------------------------------
#   Linear input units, colors: ('red', 'green'), words: ('RED','GREEN')
colors_input_layer = pnl.TransferMechanism(
    size=3,
    function=pnl.Linear,
    name='COLORS_INPUT'
)

words_input_layer = pnl.TransferMechanism(
    size=3,
    function=pnl.Linear,
    name='WORDS_INPUT'
)

# NOTE: should the task input size be different? Since the actual tasks are different in emotional stroop, 
        # What are the processes occuring? Colour naming and what? Technically emotional processing occurs, but is not the actual task.
task_input_layer = pnl.TransferMechanism(
    size=3, # NOTE: changed to 3 for the added emotion processing task. 
    function=pnl.Linear,
    name='TASK_INPUT'
)

# NOTE: Added emotional valence inputs
emotion_input_layer = pnl.TransferMechanism(
    size=3, # NOTE: positive, negative, neutral valence
    function=pnl.Linear,
    name='EMOTION_INPUT'
)

#   Task layer, tasks: ('name the color', 'read the word') 
task_layer = pnl.RecurrentTransferMechanism(
    size=3, # NOTE: changed to 3 for added emotion processing; again, questioning if this makes sense. 
    function=pnl.Logistic(),
    hetero=inhibition,
    integrator_mode=True,
    integration_rate=rate,
    name='TASK'
)

#   Hidden layer units, colors: ('red','green') words: ('RED','GREEN')
colors_hidden_layer = pnl.RecurrentTransferMechanism(
    size=3,
    function=pnl.Logistic(x_0=bias),
    integrator_mode=True,
    hetero=inhibition,
    # noise=pnl.NormalDist(mean=0.0, standard_deviation=.0),
    integration_rate=rate,  # cohen-huston text says 0.01
    name='COLORS HIDDEN'
)

words_hidden_layer = pnl.RecurrentTransferMechanism(
    size=3,
    function=pnl.Logistic(x_0=bias),
    hetero=inhibition,
    integrator_mode=True,
    # noise=pnl.NormalDist(mean=0.0, standard_deviation=.05),
    integration_rate=rate,
    name='WORDS HIDDEN'
)

# NOTE: Added emotion hidden layer
emotion_hidden_layer = pnl.RecurrentTransferMechanism(
    size=3,
    function=pnl.Logistic(x_0=bias),
    hetero=inhibition,
    integrator_mode=True,
    # noise=pnl.NormalDist(mean=0.0, standard_deviation=.05),
    integration_rate=rate,
    name='EMOTION HIDDEN'
)

#   Response layer, responses: ('red', 'green'): RecurrentTransferMechanism for self inhibition matrix
response_layer = pnl.RecurrentTransferMechanism(
    size=2, # NOTE: Need to think of what this represents in emotional stroop, since they are still listing colours. What would the other outcome be?
    function=pnl.Logistic(),
    hetero=inhibition,
    integrator_mode=True,
    integration_rate=rate,
    name='RESPONSE'
)

# Log mechanisms ------------------------------------------------------------------------------------------------------
task_layer.set_log_conditions('value')
colors_hidden_layer.set_log_conditions('value')
words_hidden_layer.set_log_conditions('value')
emotion_hidden_layer.set_log_conditions('value') # NOTE: I have no idea what this does
response_layer.set_log_conditions('value')

# Connect mechanisms --------------------------------------------------------------------------------------------------
# (note that response layer projections are set to all zero first for initialization
# input weights
color_input_weights = pnl.MappingProjection(
    matrix=np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
)

word_input_weights = pnl.MappingProjection(
    matrix=np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
)

# NOTE: added emotion weights
emotion_input_weights = pnl.MappingProjection(
    matrix=np.array([
        [1.0, 0.0, 0.0], # positive
        [0.0, 1.0, 0.0], # negative
        [0.0, 0.0, 0.0]  # neutral
    ])
)

task_input_weights = pnl.MappingProjection(
    matrix=np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0] # added third emotion layer
    ])
)

# task weights
# NOTE: I believe the 4's here represent the biases of each node. 
color_task_weights = pnl.MappingProjection(
    matrix=np.array([
        [4.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [4.0, 0.0, 0.0] # third column is emotion layer
    ])
)

task_color_weights = pnl.MappingProjection(
    matrix=np.array([
        [4.0, 4.0, 4.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0] # third row is emotion layer
    ])
)

word_task_weights = pnl.MappingProjection(
    matrix=np.array([
        [0.0, 4.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 4.0, 0.0]
    ])
)
# NOTE: changing column 3 values in here wouldn't make sense since this would mean that the input of colour/word affect emotion processing. In theory, the colour red could affect emotional processing, or reading the word red could too, but this seems to be beyond the scope of the project. for now, we have no interaction going from other systems onto emotion. Only interactions from emotion onto other systems.

task_word_weights = pnl.MappingProjection(
    matrix=np.array([
        [0.0, 0.0, 0.0],
        [4.0, 4.0, 4.0],
        [0.0, 0.0, 0.0]
    ])
)

# rows = nodes (positive, negative, neutral, columns = task (colour, word, emotion)
emotion_task_weights = pnl.MappingProjection(
    matrix=np.array([
        [0.0, 0.0, 4.0],
        [0.0, 0.0, 4.0], # Assume that negative node has a stronger bias
        [0.0, 0.0, 4.0] # in theory, emotion leads to slowing of other processes in terms of rumination
    ])                  # this could lead to slower processing of both colour naming and word reading (columns 1 and 2 respectively)
)
# adding this inhibitory effect actually slows down the entire system more. So even though neutral doesn't affect the system, it gets slowed down due to slowed colour and word reading. 

task_emotion_weights = pnl.MappingProjection(
    matrix=np.array([
        [0.0, 0.0, 0.0], 
        [0.0, 0.0, 0.0], 
        [4.0, 4.0, 4.0]
    ])
)

# response weights
response_color_weights = pnl.MappingProjection(
    matrix=np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
)

response_word_weights = pnl.MappingProjection(
    matrix=np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
)

response_emotion_weights = pnl.MappingProjection(
    matrix=np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
        # NOTE: the word and colour have no return value from the response. Does that make sense for emotion?
    ])
)

color_response_weights = pnl.MappingProjection(
    matrix=np.array([
        [1.5, 0.0],
        [0.0, 1.5],
        [0.0, 0.0]
    ])
)
word_response_weights = pnl.MappingProjection(
    matrix=np.array([
        [2.5, 0.0], # this would be red to red. This would be red to green (so rows = own node, columns = task node... duh that's why it's 3x2)
        [0.0, 2.5], # this would be green to green
        [0.0, 0.0]
    ])
)

# zeroes because emotion doesn't have a corresponding response node; could be for future research
emotion_response_weights = pnl.MappingProjection(
    matrix=np.array([
        # Our hypothesis predicts that both positive and negative cause interference.
        # seeing an emotionally valenced word would lead to slow down effects. In this case, we assume it inhibits other pathways 
        # Assumes that negative words cause greater interference
        # rows = positive, negative, neutral, columns = red, green
        [-1.5, -1.5],
        [-2.5, -2.5], 
        [0.0, 0.0] 
        
        # in theory, processing the emotional valence could help get the correct response. But that is hard to say. I've added values in the first column to represent this potential noise. 
        # Increasing the magnitudes leads to larger differences. But determining the actual strength of the pathway is beyond the scope of this project. 
        # [0.5, 1.5],
        # [0.5, 2.5], 
        # [0.0, 0.0] 
    ])
)

#
# Create pathways -----------------------------------------------------------------------------------------------------
# response pathways
color_response_process_1 = pnl.Pathway(
    pathway=[
        colors_input_layer,
        color_input_weights,
        colors_hidden_layer,
        color_response_weights,
        response_layer
    ],
    name='COLORS_RESPONSE_PROCESS_1'
)

color_response_process_2 = pnl.Pathway(
    pathway=[
        response_layer,
        response_color_weights,
        colors_hidden_layer
    ],
    name='COLORS_RESPONSE_PROCESS_2'
)

word_response_process_1 = pnl.Pathway(
    pathway=[
        words_input_layer,
        word_input_weights,
        words_hidden_layer,
        word_response_weights,
        response_layer
    ],
    name='WORDS_RESPONSE_PROCESS_1'
)

word_response_process_2 = pnl.Pathway(
    pathway=[
        (response_layer, pnl.NodeRole.OUTPUT), # this is related to logging the output results... probably
        response_word_weights,
        words_hidden_layer
    ],
    name='WORDS_RESPONSE_PROCESS_2'
)

# NOTE: emotion layers
emotion_response_process_1 = pnl.Pathway(
    pathway=[
        emotion_input_layer,
        emotion_input_weights,
        emotion_hidden_layer,
        emotion_response_weights,
        response_layer # NOTE: there was a comma at the end here, but I removed it. See if model breaks
    ],
    name='EMOTION_RESPONSE_PROCESS_1'
)

# NOTE:
emotion_response_process_2 = pnl.Pathway(
    pathway=[
        response_layer,
        response_emotion_weights,
        emotion_hidden_layer
    ],
    name='EMOTION_RESPONSE_PROCESS_2'
)

# task pathways
task_color_response_process_1 = pnl.Pathway(
    pathway=[
        task_input_layer,
        task_input_weights,
        task_layer,
        task_color_weights,
        colors_hidden_layer])

task_color_response_process_2 = pnl.Pathway(
    pathway=[
        colors_hidden_layer,
        color_task_weights,
        task_layer])

task_word_response_process_1 = pnl.Pathway(
    pathway=[
        task_input_layer,
        task_layer,
        task_word_weights,
        words_hidden_layer])

task_word_response_process_2 = pnl.Pathway(
    pathway=[
        words_hidden_layer,
        word_task_weights,
        task_layer])

task_emotion_response_process_1 = pnl.Pathway(
    pathway=[
        task_input_layer,
        task_input_weights,
        task_layer,
        task_emotion_weights,
        emotion_hidden_layer])

task_emotion_response_process_2 = pnl.Pathway(
    pathway=[
        emotion_hidden_layer,
        emotion_task_weights,
        task_layer])

# Create Composition --------------------------------------------------------------------------------------------------
Bidirectional_Stroop = pnl.Composition(
    pathways=[
        color_response_process_1,
        word_response_process_1,
        emotion_response_process_1,
        task_color_response_process_1,
        task_word_response_process_1,
        task_emotion_response_process_1,
        color_response_process_2,
        word_response_process_2,
        emotion_response_process_2,
        task_color_response_process_2,
        task_word_response_process_2,
        task_emotion_response_process_2
    ],
    reinitialize_mechanisms_when=pnl.Never(),
    name='Bidirectional Stroop Model'
)

input_dict = {colors_input_layer: [0, 0, 0],
              words_input_layer: [0, 0, 0],
              emotion_input_layer: [0, 0, 0], # NOTE: added emotion input to input dict. 
              task_input_layer: [0, 1, 0]} # NOTE: added extra emotion task input.
                # I believe the 1 would indiciate what task is being done. With color first, word second, emotion third
                # Similarly, I believe that the other inputs layers correspond to the condition (ex: negative, positive, congruent,incongruent)
print("\n\n\n\n")
print(Bidirectional_Stroop.run(inputs=input_dict))

for node in Bidirectional_Stroop.mechanisms:
    print(node.name, " Value: ", node.get_output_values(Bidirectional_Stroop))


# # LOGGING:
colors_hidden_layer.set_log_conditions('value')
words_hidden_layer.set_log_conditions('value')
emotion_hidden_layer.set_log_conditions('value') # NOTE: added emotion layer logging.

# Create threshold function -------------------------------------------------------------------------------------------

terminate_trial = {
    pnl.TimeScale.TRIAL: pnl.Or(
        pnl.Threshold(response_layer, 'value', threshold, '>=', (0, 0)),
        pnl.Threshold(response_layer, 'value', threshold, '>=', (0, 1)),
    )
}

# Create test trials function -----------------------------------------------------------------------------------------
# a BLUE word input is [1,0] to words_input_layer and GREEN word is [0,1]
# a blue color input is [1,0] to colors_input_layer and green color is [0,1]
# a color-naming trial is [1,0] to task_layer and a word-reading trial is [0,1]

def trial_dict(red_color, green_color, neutral_color, red_word, green_word, neutral_word, positive_emotion, negative_emotion, neutral_emotion, CN, WR, EP):
# CN = colour naming, WR = word reading, EP = emotion processing. 
    trialdict = {
        colors_input_layer: [red_color, green_color, neutral_color],
        words_input_layer: [red_word, green_word, neutral_word],
        emotion_input_layer: [positive_emotion, negative_emotion, neutral_emotion],
        task_input_layer: [CN, WR, EP]
    }
    return trialdict

# NOTE: I realize now that there was no point in creating an emotion processing variable, since we are measuring color naming trials the person will never purposely do emotion processing.

# Define initialization trials separately
# order: red_color, green_color, neutral_color, red_word, green_word, neutral_word, positive_emotion, negative_emotion, neutral_emotion, CN, WR, EP
CN_initialize_input = trial_dict(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0)
WR_initialize_input = trial_dict(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0)
EP_initialize_input = trial_dict(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)

# red_color, green_color, neutral_color, red_word, green_word, neutral_word, positive_emotion, negative_emotion, neutral_emotion, CN, WR, EP
CN_congruent_trial_input =   trial_dict(1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0) # red_colour + red_word
CN_incongruent_trial_input = trial_dict(1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0) # red_colour + green_word
CN_control_trial_input =     trial_dict(1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0) # red_colour + no word (?)

WR_congruent_trial_input =   trial_dict(1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0) # red_color + red_word
WR_incongruent_trial_input = trial_dict(1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0) # red_colour + green_word
WR_control_trial_input =     trial_dict(0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0) # no color? + red word 

CN_positive_trial_input = trial_dict(1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)
CN_negative_trial_input = trial_dict(1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0)
CN_neutral_trial_input =  trial_dict(1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0)

Stimulus = [[CN_initialize_input, CN_control_trial_input],
            [CN_initialize_input, CN_incongruent_trial_input],
            [CN_initialize_input, CN_congruent_trial_input]]

Stimulus2 = [[WR_initialize_input, WR_control_trial_input],
             [WR_initialize_input, WR_incongruent_trial_input],
             [WR_initialize_input, WR_control_trial_input]]
             
Stimulus3 = [
                [CN_initialize_input, CN_negative_trial_input],
                [CN_initialize_input, CN_neutral_trial_input],
                [CN_initialize_input, CN_positive_trial_input]
            ]

# Create third stimulus? Technically we would only have colour naming trials to begin with. So I guess a third stimulus except it would be a colour naming one with emotional words activated, but not the actual task node.
    # It would be like the control CN task, but instead the emotional words are used.

conditions = 3
response_all = []
response_all2 = []
# # Run color naming trials ----------------------------------------------------------------------------------------------
# for cond in range(conditions):
#     response_color_weights.parameters.matrix.set(
#         np.array([
#             [0.0, 0.0, 0.0],
#             [0.0, 0.0, 0.0]
#         ]), Bidirectional_Stroop
#     )
#     response_word_weights.parameters.matrix.set(
#         np.array([
#             [0.0, 0.0, 0.0],
#             [0.0, 0.0, 0.0]
#         ]), Bidirectional_Stroop
#     )
#     # NOTE: Added response_emotion weights
#     response_emotion_weights.parameters.matrix.set(
#         np.array([
#             [0.0, 0.0, 0.0],
#             [0.0, 0.0, 0.0]
#         ]), Bidirectional_Stroop
#     )
#     Bidirectional_Stroop.run(inputs=Stimulus[cond][0], num_trials=settle_trials)

#     response_color_weights.parameters.matrix.set(
#         np.array([
#             [1.5, 0.0, 0.0],
#             [0.0, 1.5, 0.0]
#         ]), Bidirectional_Stroop
#     )
#     response_word_weights.parameters.matrix.set(
#         np.array([
#             [2.5, 0.0, 0.0],
#             [0.0, 2.5, 0.0]
#         ]), Bidirectional_Stroop
#     )

#     # NOTE: Added response_emotion weights
#     # Technically, the inputs here should be 0 for these standard Stroop trials. So the actual weights don't matter here.
#     # However, the magnitude of these weights would essentially signify the pathway strength. So it's hard to say for now
#     response_emotion_weights.parameters.matrix.set(
#         np.array([
#             [1.5, 0.0, 0.0],
#             [0.0, 1.5, 0.0]
#         ]), Bidirectional_Stroop
#     )

#     Bidirectional_Stroop.run(inputs=Stimulus[cond][1], termination_processing=terminate_trial)

#     # Store values from run -----------------------------------------------------------------------------------------------
#     B_S = Bidirectional_Stroop.name
#     r = response_layer.log.nparray_dictionary('value')       # Log response output from special logistic function
#     rr = r[B_S]['value']
#     n_r = rr.shape[0]
#     rrr = rr.reshape(n_r, 2) # NOTE: I believe this stays the same, since the output should be the same, despite having 3 inputs now. 
#     response_all.append(rrr)  # .shape[0])
#     response_all2.append(rrr.shape[0]) # NOTE: I really wish I knew what was happening here. 

#     # Clear log & reset ----------------------------------------------------------------------------------------
#     response_layer.log.clear_entries()
#     colors_hidden_layer.log.clear_entries()
#     words_hidden_layer.log.clear_entries()
#     emotion_hidden_layer.log.clear_entries() # NOTE: Clear emotion hidden layer logs
#     task_layer.log.clear_entries()
#     colors_hidden_layer.reset([[0, 0, 0]]) 
#     words_hidden_layer.reset([[0, 0, 0]])
#     emotion_hidden_layer.reset([[0, 0, 0]])
#     response_layer.reset([[0, 0]])
#     task_layer.reset([[0, 0, 0]]) # NOTE: task layer reset needs 3 nodes now.
#     print('response_all: ', response_all)
#     print('first trials')

# # Run color naming trials ----------------------------------------------------------------------------------------------
# response_all3 = []
# response_all4 = []
# print('made the next responses')
# for cond in range(conditions):
#     response_color_weights.parameters.matrix.set(
#         np.array([
#             [0.0, 0.0, 0.0],
#             [0.0, 0.0, 0.0]
#         ]), Bidirectional_Stroop
#     )
#     response_word_weights.parameters.matrix.set(
#         np.array([
#             [0.0, 0.0, 0.0],
#             [0.0, 0.0, 0.0]
#         ]), Bidirectional_Stroop
#     )
#     # NOTE: Added response_emotion weights
#     response_emotion_weights.parameters.matrix.set(
#         np.array([
#             [0.0, 0.0, 0.0],
#             [0.0, 0.0, 0.0]
#         ]), Bidirectional_Stroop
#     )
#     Bidirectional_Stroop.run(inputs=Stimulus2[cond][0], num_trials=settle_trials)

#     response_color_weights.parameters.matrix.set(
#         np.array([
#             [1.5, 0.0, 0.0],
#             [0.0, 1.5, 0.0]
#         ]), Bidirectional_Stroop
#     )
#     response_word_weights.parameters.matrix.set(
#         np.array([
#             [2.5, 0.0, 0.0],
#             [0.0, 2.5, 0.0]
#         ]), Bidirectional_Stroop
#     )

#     # NOTE: Added response_emotion weights
#     response_emotion_weights.parameters.matrix.set(
#         np.array([
#             [1.5, 0.0, 0.0],
#             [0.0, 1.5, 0.0]
#         ]), Bidirectional_Stroop
#     )

#     Bidirectional_Stroop.run(inputs=Stimulus2[cond][1], termination_processing=terminate_trial)

#     # Store values from run -----------------------------------------------------------------------------------------------
#     r2 = response_layer.log.nparray_dictionary('value')       # Log response output from special logistic function
#     rr2 = r2[Bidirectional_Stroop.name]['value']
#     n_r2 = rr2.shape[0]
#     rrr2 = rr2.reshape(n_r2, 2)
#     response_all3.append(rrr2)  # .shape[0])
#     response_all4.append(rrr2.shape[0])

#     # Clear log & reset ----------------------------------------------------------------------------------------
#     response_layer.log.clear_entries()
#     colors_hidden_layer.log.clear_entries()
#     words_hidden_layer.log.clear_entries()
#     emotion_hidden_layer.log.clear_entries() # NOTE: Clear emotion hidden layer logs
#     task_layer.log.clear_entries()
#     colors_hidden_layer.reset([[0, 0, 0]])
#     words_hidden_layer.reset([[0, 0, 0]])
#     emotion_hidden_layer.reset([[0, 0, 0]])
#     response_layer.reset([[0, 0]])
#     task_layer.reset([[0, 0, 0]]) # NOTE: again, 3 nodes now
#     print('response_all: ', response_all3)
#     print('got to second trials')

# Run color naming with emotion  of a high ruminator ----------------------------------------------------------------------------------------------
response_all5 = []
response_all6 = []
print('made the next responses')
for cond in range(conditions):
    response_color_weights.parameters.matrix.set(
        np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]), Bidirectional_Stroop
    )
    response_word_weights.parameters.matrix.set(
        np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]), Bidirectional_Stroop
    )
    # NOTE: Added response_emotion weights
    response_emotion_weights.parameters.matrix.set(
        np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]), Bidirectional_Stroop
    )
    Bidirectional_Stroop.run(inputs=Stimulus3[cond][0], num_trials=settle_trials)

    response_color_weights.parameters.matrix.set(
        np.array([
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0]
        ]), Bidirectional_Stroop
    )
    response_word_weights.parameters.matrix.set(
        np.array([
            [2.5, 0.0, 0.0],
            [0.0, 2.5, 0.0]
        ]), Bidirectional_Stroop
    )

        # [0.5, 2.5],
        # [0.5, 3.0], # this configuration makes it so that. In essence, both positive and negative cause interference from getting the correct colour
        # [0.0, 0.0] # in theory, processing the emotional valence could help get the correct response. But that is hard to say. So I've added values in the first column to represent this potential noise. 
    
    # NOTE: Added response_emotion weights
    response_emotion_weights.parameters.matrix.set(
        np.array([
            # assumes that getting it right or wrong has no effect on the input nodes. Since saying either colour shouldn't lead to changes in emotional processing... right? 
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0] 

            # In theory, getting it right or wrong could have a small effect on a person's emotional processing. Like getting it wrong would make them feel worse or overwhelmed. 
            # assumes that the effect of the response nodes on the input nodes is equal between valence and weaker than the reverse connection. 
            # [0.5, 0.5, 0.0],
            # [0.5, 0.5, 0.0] 
          
        ]), Bidirectional_Stroop
    )

    Bidirectional_Stroop.run(inputs=Stimulus3[cond][1], termination_processing=terminate_trial)

    # Store values from run -----------------------------------------------------------------------------------------------
    r3 = response_layer.log.nparray_dictionary('value')       # Log response output from special logistic function
    rr3 = r3[Bidirectional_Stroop.name]['value']
    n_r3 = rr3.shape[0]
    rrr3 = rr3.reshape(n_r3, 2)
    response_all5.append(rrr3)  # .shape[0])
    response_all6.append(rrr3.shape[0])

    # Clear log & reset ----------------------------------------------------------------------------------------
    response_layer.log.clear_entries()
    colors_hidden_layer.log.clear_entries()
    words_hidden_layer.log.clear_entries()
    emotion_hidden_layer.log.clear_entries() # NOTE: Clear emotion hidden layer logs
    task_layer.log.clear_entries()
    colors_hidden_layer.reset([[0, 0, 0]])
    words_hidden_layer.reset([[0, 0, 0]])
    emotion_hidden_layer.reset([[0, 0, 0]])
    response_layer.reset([[0, 0]])
    task_layer.reset([[0, 0, 0]]) # NOTE: again, 3 nodes now
    print('response_all: ', response_all5)
    print('got to third trials')

############################################################################################
# Run color naming with emotion of a low ruminator ----------------------------------------------------------------------------------------------
response_all7 = []
response_all8 = []
print('made the next responses')
for cond in range(conditions):
    response_color_weights.parameters.matrix.set(
        np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]), Bidirectional_Stroop
    )
    response_word_weights.parameters.matrix.set(
        np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]), Bidirectional_Stroop
    )
    # NOTE: Added response_emotion weights
    response_emotion_weights.parameters.matrix.set(
        np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]), Bidirectional_Stroop
    )
    Bidirectional_Stroop.run(inputs=Stimulus3[cond][0], num_trials=settle_trials)

    response_color_weights.parameters.matrix.set(
        np.array([
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0]
        ]), Bidirectional_Stroop
    )
    response_word_weights.parameters.matrix.set(
        np.array([
            [2.5, 0.0, 0.0],
            [0.0, 2.5, 0.0]
        ]), Bidirectional_Stroop
    )

        # [0.5, 2.5],
        # [0.5, 3.0], # this configuration makes it so that. In essence, both positive and negative cause interference from getting the correct colour
        # [0.0, 0.0] # in theory, processing the emotional valence could help get the correct response. But that is hard to say. So I've added values in the first column to represent this potential noise. 
    
    # NOTE: Added response_emotion weights
    response_emotion_weights.parameters.matrix.set(
        np.array([
            # assumes that getting it right or wrong has no effect on the input nodes. Since saying either colour shouldn't lead to changes in emotional processing... right? 
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0] 

            # In theory, getting it right or wrong could have a small effect on a person's emotional processing. Like getting it wrong would make them feel worse or overwhelmed. 
            # assumes that the effect of the response nodes on the input nodes is equal between valence and weaker than the reverse connection. 
            # [0.5, 0.5, 0.0],
            # [0.5, 0.5, 0.0] 
          
        ]), Bidirectional_Stroop
    )

    # NOTE: change the input node effects too
    emotion_response_weights.parameters.matrix.set(
        np.array([
            [-0.25, -0.25],
            [-0.5, -0.5], # remove the interference caused by emotion
            [0.0, 0.0] 
        ]), Bidirectional_Stroop
    )
    

    Bidirectional_Stroop.run(inputs=Stimulus3[cond][1], termination_processing=terminate_trial)

    # Store values from run -----------------------------------------------------------------------------------------------
    r4 = response_layer.log.nparray_dictionary('value')       # Log response output from special logistic function
    rr4 = r4[Bidirectional_Stroop.name]['value']
    n_r4 = rr4.shape[0]
    rrr4 = rr4.reshape(n_r4, 2)
    response_all7.append(rrr4)  # .shape[0])
    response_all8.append(rrr4.shape[0])

    # Clear log & reset ----------------------------------------------------------------------------------------
    response_layer.log.clear_entries()
    colors_hidden_layer.log.clear_entries()
    words_hidden_layer.log.clear_entries()
    emotion_hidden_layer.log.clear_entries() # NOTE: Clear emotion hidden layer logs
    task_layer.log.clear_entries()
    colors_hidden_layer.reset([[0, 0, 0]])
    words_hidden_layer.reset([[0, 0, 0]])
    emotion_hidden_layer.reset([[0, 0, 0]])
    response_layer.reset([[0, 0]])
    task_layer.reset([[0, 0, 0]]) # NOTE: again, 3 nodes now
    print('response_all: ', response_all7)
    print('got to fourth trials')




print('now we plot')
if args.enable_plot:
    import matplotlib.pyplot as plt
    # # Plot results --------------------------------------------------------------------------------------------------------
    # # First, plot response layer activity for whole run
    # plt.figure()
    # # color naming plot
    # plt.plot(response_all[0])
    # plt.plot(response_all[1])
    # plt.plot(response_all[2])
    # # word reading plot
    # plt.plot(response_all3[0])
    # plt.plot(response_all3[1])
    # plt.plot(response_all3[2])
    # plt.show(block=not pnl._called_from_pytest)
    # # Second, plot regression plot
    # # regression
    # reg = np.dot(response_all2, 5) + 115
    # reg2 = np.dot(response_all4, 5) + 115
    # plt.figure()

    # plt.plot(reg, '-s')  # plot color naming
    # plt.plot(reg2, '-or')  # plot word reading
    
    # plt.title('GRAIN MODEL with bidirectional weights')
    # plt.legend(['color naming', 'word reading'])
    # plt.xticks(np.arange(3), ('control', 'incongruent', 'congruent'))
    # plt.ylabel('reaction time in ms')
    # plt.show(block=not pnl._called_from_pytest)

    # Show emotional graph
    reg3 = np.dot(response_all6, 5) + 115
    plt.plot(reg3, '-x')
    plt.xlabel('Valence')
    plt.title('Simulated GRAIN data')
    plt.xticks(np.arange(3), ('Negative', 'Neutral', 'Positive'))
    plt.ylabel('Reaction Time (ms)')
    plt.show(block=not pnl._called_from_pytest)

    # show other emotion graph of a low ruminator
    reg4 = np.dot(response_all8, 5) + 115
    plt.plot(reg4, '-x')
    plt.xlabel('Valence')
    plt.title('Simulated GRAIN data')
    plt.xticks(np.arange(3), ('Negative', 'Neutral', 'Positive'))
    plt.ylabel('Reaction Time (ms)')
    plt.show(block=not pnl._called_from_pytest)