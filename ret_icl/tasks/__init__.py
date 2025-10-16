
def get_task(name, file='24_train.csv'):
    if name == 'game24':
        from src.ret_icl.tasks.game24 import Game24Task
        return Game24Task(file=file)
    # elif name == 'text':
    #     from tot.tasks.text import TextTask
    #     return TextTask()
    # elif name == 'crosswords':
    #     from tot.tasks.crosswords import MiniCrosswordsTask
    #     return MiniCrosswordsTask()
    else:
        raise NotImplementedError