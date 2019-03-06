import argparse


def squareroot(x):
    """Calculate square root"""
    return x**(0.5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates square root of given number')

    parser.add_argument('-a','--sayi', '--number', required=True,
                        help='number to be squarerooted')
    args = vars(parser.parse_args())

    print(squareroot(args['a']))
    # a = resimOku(resimIsmi)
    # yenidenBoyutlandirilmisA = yenidenBoyutlandir(a)
    #
    # iwrite(cikisIsmi ,yenidenBoyutlandirilmisA)
