import pipeline_utils as put

def main():
    print("Cancer Dataset Flip")
    put.test_posion_1(posionType='FLIP', percent=0.1, number=10, dataset='cancer', mode = 'random', style="SPLIT")
    
    print("Cancer Dataset Inject")
    put.test_posion_1(posionType='INJECT', percent=0.1, number=10, dataset='cancer', mode = 'random', style = "SPLIT")

    print("Cancer Dataset Tamper")
    put.test_posion_1(posionType='TAMPER', percent=0.1, number=10, dataset='cancer', mode = 'random', style= 'SPLIT')

    # print("Cancer Dataset")
    # put.test_posion(posionType='INJECT', percent=0.1, number = 10, dataset='cancer',mode = 'random')
    # print("Machine Dataset")
    # put.test_posion(posionType='INJECT', percent=0.1, number = 10, dataset='machine',mode = 'random')
    # print("Loan Dataset")
    # put.test_posion(posionType='INJECT', percent=0.1, number = 10, dataset='loan',mode = 'random')

if __name__ == '__main__': 
    main()