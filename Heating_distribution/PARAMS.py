from PARAMS_SN import TARGET


def DQN_PARAMS(num=TARGET):
    res={
    1:{
    'yes':1,
    },
    }
    return res.get(num,None)
