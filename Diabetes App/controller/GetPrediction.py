import numpy as np 
def GetPredection(model, Inputs: list):
    result = {
        0: "The analysis shows that you are not at risk for diabetes. Keep up the good work with your health and wellness habits.",
        1: "The results suggest that you may have diabetes. It's important to take proactive steps and seek medical guidance for management and support."
    }
    
   
    inputs_array = np.array(Inputs).reshape(1, -1)  
    
    predictions = model.predict(inputs_array)[0] 
    
    return result[predictions]
