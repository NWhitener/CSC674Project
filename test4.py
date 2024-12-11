import poison_utils as pu 
import preprocessing_utils as ppu 
import pipeline_utils as pppu 
import model_utils as mut 



'''
Heart Data Collection Results Final Version
'''


# pppu.test_posion_1(posionType="TAMPER", percent=0.1, number = 10, dataset= 'HEART',mode = "DISTRIBUTION", style = "SPLIT")
# pppu.test_posion_1(posionType="INJECT", percent=0.1, number = 10, dataset= 'HEART',mode = "DISTRIBUTION", style = "SPLIT")
pppu.test_posion_1(posionType="FLIP", percent=0.1, number = 10, dataset= 'HEART',mode = "DISTRIBUTION", style = "SPLIT")


'''
Cancer Data Collection Results Final Version
'''
# pppu.test_posion_1(posionType="TAMPER", percent=0.1, number = 10, dataset= 'CANCER',mode = "DISTRIBUTION", style = "SPLIT")
# pppu.test_posion_1(posionType="INJECT", percent=0.1, number = 10, dataset= 'CANCER',mode = "DISTRIBUTION", style = "SPLIT")
# pppu.test_posion_1(posionType="FLIP", percent=0.1, number = 10, dataset= 'CANCER',mode = "DISTRIBUTION", style = "SPLIT")