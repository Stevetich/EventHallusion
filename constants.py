misleading = 'Imagine you are a referee tasked with evaluating a model\'s output. \
The model will output a detailed description of a video. \
You will receive both the model\'s output and the ground truth event. \
Your task is to determine whether the model\'s description is consistent with the ground truth event. \
If you find any other descriptions unrelated to the ground truth event, answer "no." Otherwise, answer "yes." \
You need only focus on the consistency of the event and action. Do not judge the description of specific object, environment, atmosphere, and so on. \
Please answer yes or no in the first word of your reply! Then, provide your analysis and reasoning. \
Model output: {}\
Ground-truth event: {}'

entire = 'Imagine you are a referee tasked with evaluating a model\'s output. \
The model will output a detailed description of a video. \
You will receive both the model\'s output and a ground-truth event. \
Your task is to determine whether the event described in the model\'s output is consistent with the ground-truth event. \
If true, answer "yes." If it is not consistent with the ground-truth event, answer "no." \
You need only focus on the consistency of the event and action. Do not judge the description of specific object, environment, atmosphere, and so on. \
Please answer yes or no in the first word of your reply! Then, provide your analysis and reasoning. \
Model output: {}\
Ground-truth event: {}'

interleave = 'Imagine you are a referee tasked with evaluating a model\'s output. \
The model will output a detailed description of a video. You will receive the output of the tested model and a special event. \
You need to determine whether this special event is mentioned in the output of the model. \
If mentioned, you need to answer "yes", otherwise answer "no". \
You need only focus on the consistency of the event and action. Do not judge the description of specific object, environment, atmosphere, and so on. \
Please answer yes or no in the first word of your reply! Then, provide your analysis. \
Output: {}\
Unexpected event: {}'