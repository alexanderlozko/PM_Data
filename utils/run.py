from utils.model import Model


def run():
    print('\n')
    print('''Train model - 1\nPredict results - 2\nInctruction for use - 3\nExit - 0''')
    answ = int(input('Your choice: '))
    print('\n')

    if answ == 2:
        person = input('Enter the number of person (01-16): ')
        predict = input('What do you want to predict? (result/mood/stress/fatigue): ')

        result = Model.predict(person, predict)

        for key in result:
            print(key, ':', '\n', result[key])

        return run()

    elif answ == 1:
        person = input('Enter the number of person (01-16): ')
        predict = input('What do you want to predict? (result/mood/stress/fatigue): ')

        Model.prepare_model(person, predict)

        return run()

    elif answ == 3:

        print('''
        FAMILIARIZATION WITH DEMO PRODUCT
        
        First of all, you need to train the model. Press 1. Next, you select the number of the person on whose data you want to train the model.
        After choosing what you want to predict, you can choose: "result" is a change in weight, "mood" is the level of mood, "stress" is the level of stress, "fatigue" is the level of fatigue.
        After training, you get a plot with evaluating the accuracy of the classifiers. And the best classifier is automatically selected and saved. To save the model, you need to stop and restart the script. 
        After a new start, you can press 2. Enter the person number and category you want to predict again. 
        Further, the model will use test data, which were divided at the time of training, to predict and display the results.
        ''')
        run()

    elif answ == 0:
        print('Bye!')

if __name__ == '__main__':
    run()