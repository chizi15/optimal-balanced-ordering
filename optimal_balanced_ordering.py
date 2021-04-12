import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def opt_baln_order(df_all):
    """
    本函数实现的功能：
    找到各个供应商下所有单品在当前状态下的最优平衡周转天数，低于该天数的商品一律通过追加补货量的方式拉齐到该平衡周转天数，高于该天数的商品则不动；
    且使各个单品的总补货量为自身件规格的整数倍；且使供应商下所有单品的最终补货件数之和恰好等于该供应商的最小起订件数；
    且尽可能减少最优平衡周转天数的搜索次数，使计算速度加快。
    :param：传入一个或多个供应商的补货相关信息的df，包括的数据维度有['code', 'provider', 'mini-order-unit',
    'unit', 'stock', 'arriving', 'ordering', 'dms', 'retention-days', 'already-unit']。
    :return：返回一个或多个供应商的最终补货信息的df，在传入的df_all的基础上，增加了['final-ordering', 'final-unit', 'final-ret-days']三个维度。
    """

    if df_all.isna().any().any():
        raise Exception('原数据中含有空值，请检查')
    df_all = pd.concat([df_all, pd.DataFrame(columns=list('ABC'))]).fillna(0)
    df_all.rename(columns={"A": "final-ordering", "B": "final-unit", 'C': 'final-ret-days'}, inplace=True)
    for i in range(len(set(df_all['provider']))):
        df_ori = df_all[df_all['provider'] == list(set(df_all['provider']))[i]]
        df = df_ori.copy()  # 使用copy方法另存一份df，否则df会和df_ori使用同一个存储地址，则更改会同时起作用
        if sum(df['already-unit']) >= pd.Series(list(set(df['mini-order-unit'])))[0]:
            for _ in df['already-unit'].index:
                df_all['final-ordering'][df_all['final-ordering'].index == _] = df['ordering'][df['ordering'].index == _]
                df_all['final-unit'][df_all['final-unit'].index == _] = df['already-unit'][df['already-unit'].index == _]
                # 使展示的最终周转天数的小数位数与原始数据中日均销售的小数位数相同
                df_all['final-ret-days'][df_all['final-ret-days'].index == _] = \
                    round((df['stock'] + df['arriving'] + df['ordering']) / df['dms'], 3)
            print('provider %s下的单品无需追加补货，因其已定件数总量不小于最小起订量' % int(list(set(df['provider']))[-1]))
            continue  # 跳过后续语句，进行下一次"for i in range(len(set(df_all['provider'])))"循环

        # 将df中日均销量<=0的行剔除，以保证后续计算周转天数的准确性。当日均销量为0时，会使计算周转天数为空，此时原数据中对应周转天数被置为0，
        # 表示极度缺货，而实际可能是很长时间没有销售；对于这种情况，则不进行追加补货，以防很久都卖不出去。
        df['dms'] = df['dms'][df['dms'] > 0]
        df.dropna(inplace=True)
        print('\n', '第%s个供应商编码：' % (i+1), int(list(set(df['provider']))[-1]), '\n', '该供应商下日均销量大于0的单品数：', len(df), '\n')

        if len(df) > 1:
            # 周转天数和已定件数是非独立变量，周转天数由（库存 + 在途 + ordering） / dms（向下取整）得到，
            # 已定件数由计算订货量 / 商品件规格（须为整数）得到。其中因为周转天数向下取整，在计算过程中会产生传播误差，
            # 所以公式中不应使用取整后的周转天数，而应使用准确的小数形式周转天数。
            order_ori, T_ori, unit_ori = df['ordering'], df['retention-days'], df['unit']
            T = (df['stock'] + df['arriving'] + order_ori) / df['dms']  # 第一次未截断前，商品的精确周转天数
            if sum(T - T_ori >= 1) > 0:
                raise Exception('原始数据中周转天数向下取整时有误')
            critic = T.max()

            T_u = (df[T > critic]['stock'] + df[T > critic]['arriving'] + order_ori[T > critic]) / df[T > critic]['dms']
            df_trunc = df[T <= critic]  # 若因df['dms']中存在0，而使T中存在nan，则T <= critic会被判定为false，则会被排除掉而不进入df_trunc
            mini = pd.Series(list(set(df_trunc['mini-order-unit'])))[0]  # set取集合，即只取样本中的不同元素；再将元素作为值取出，方便后续计算
            unit = np.longdouble(df_trunc['unit'])  # 商品件规格unit应转为计算机最高存储位数的数据类型，如float128，
            # 否则当一个供应商下需分配的单品过多时，prod()无法算出连乘项，则会返回0；而使用循环会报"overflow error"也无法算出。
            stock, arri, order, dms = df_trunc['stock'], df_trunc['arriving'], df_trunc['ordering'], df_trunc['dms']
            T_d = (stock + arri + order) / dms
            alr_uni = order_ori / unit_ori  # 未被截断前各单品的已定件数

            A = stock + arri + order  # 数量, series

            B = []
            for _ in range(len(unit)):
                unit_m = np.ma.array(unit, mask=False)
                unit_m.mask[_] = True
                B.append(unit_m.prod())
            B = np.array(B)  # array of longdouble

            pai_unit = unit.prod()  # prod()对序列做连乘
            # 循环方式对序列做连乘
            # pai_unit = 1
            # for i in range(len(unit)):
            #     pai_unit = pai_unit*unit[i]
            if not (sum(pai_unit / B - unit) < 1e-10):
                raise Exception('原始数据中商品件规格有误，导致连乘项B[i]计算有误；或截断后单品个数小于2')

            C = (mini - sum(alr_uni)) * pai_unit  # longdouble
            if C <= 0:
                raise Exception('provider %s中的单品件规格存在非正数' % int(list(set(df['provider']))[-1]))

            X = (C + np.dot(np.array(A), B)) / np.dot(np.array(dms), B)  # X是最优平衡周转天数的精确值，longdouble
            order_delta = (X - T_d) * dms  # 各单品所需增加补货的数量的精确值，series of longdouble

            j, k = 0, 0
            while sum(order_delta < 0) > 0:
                j += 1
                print('补货量增量不能为负，应当降低critic取值；这是第%s次向下二分搜索最优平衡周转天数' % j)
                critic = critic / 2  # 用二分法向下搜索最优临界值
                T_u = (df[T > critic]['stock'] + df[T > critic]['arriving'] + order_ori[T > critic]) / df[T > critic]['dms']
                df_trunc = df[T <= critic]
                unit = np.longdouble(df_trunc['unit'])
                stock, arri, order, dms = df_trunc['stock'], df_trunc['arriving'], df_trunc['ordering'], df_trunc['dms']
                T_d = (stock + arri + order) / dms

                A = stock + arri + order  # 数量, series

                B = []
                for _ in range(len(unit)):
                    unit_m = np.ma.array(unit, mask=False)
                    unit_m.mask[_] = True
                    B.append(unit_m.prod())
                B = np.array(B)  # array of longdouble

                pai_unit = unit.prod()
                if not (sum(pai_unit / B - unit) < 1e-10):
                    raise Exception('连乘项B[i]计算有误，或截断后单品个数小于2')

                C = (mini - sum(alr_uni)) * pai_unit  # longdouble

                X = (C + np.dot(np.array(A), B)) / np.dot(np.array(dms), B)  # X是最优平衡周转天数的精确值，longdouble
                order_delta = (X - T_d) * dms  # 各单品所需增加补货的数量的精确值，series of longdouble

                while len(T_u[T_u < X]) > 0:
                    k += 1
                    print('critic取值过小，筛选掉过多单品，使最优平衡周转天数过大，应增加critic取值；这是第%s次向上二分搜索最优平衡周转天数' % k)
                    critic = critic * 1.5  # 用二分法向上搜索最优临界值
                    T_u = (df[T > critic]['stock'] + df[T > critic]['arriving'] + order_ori[T > critic]) / df[T > critic]['dms']
                    df_trunc = df[T <= critic]
                    unit = np.longdouble(df_trunc['unit'])
                    stock, arri, order, dms = df_trunc['stock'], df_trunc['arriving'], df_trunc['ordering'], df_trunc['dms']
                    T_d = (stock + arri + order) / dms

                    A = stock + arri + order  # 数量, series

                    B = []
                    for _ in range(len(unit)):
                        unit_m = np.ma.array(unit, mask=False)
                        unit_m.mask[_] = True
                        B.append(unit_m.prod())
                    B = np.array(B)  # array of longdouble

                    pai_unit = unit.prod()
                    if not (sum(pai_unit / B - unit) < 1e-10):
                        raise Exception('连乘项B[i]计算有误，或截断后单品个数小于2')

                    C = (mini - sum(alr_uni)) * pai_unit  # longdouble

                    X = (C + np.dot(np.array(A), B)) / np.dot(np.array(dms), B)
                    order_delta = (X - T_d) * dms
            print('总共搜索%s次找到最优平衡周转天数' % (j+k), '\n')

            exact_order = order + order_delta  # 各单品总补货量的精确值，longdouble
            exact_unit = exact_order / unit
            ceil_unit, floor_unit = np.ceil(exact_unit), np.floor(exact_unit)
            # 理论上应当用np.ceil()，对各单品的最终补货件数，向上取整，为了满足供应商最小订货件数的要求；
            # 但实际上使用np.floor()更好，因为补货件数少的单品通常dms较低，如果向上取整多补零点几件虽然能满足供应商最小订货件数的要求，
            # 但会较大幅度地增加这种低销量单品的周转天数，对实际销售不利。所以采用np.floor()，如果最终订货件数不足，则去增加销量最大单品的订货件数，
            # 也不会使其增加多少周转天数。sum(exact_order / unit)不会精确等于最小订货件数mini，有时甚至有一定程度的偏差，是因为传播误差的存在。
            # 对于无理数或无限循环小数，当小数点前的位数越多，由于计算机存储位数的有限性，会截断小数点后越大的位数；
            # 当经过多次运算，特别是乘除法、开根号等，以及参与运算的数值越大，则会产生越大的传播误差。但通常情况下，传播误差很小，对实际结果基本不影响。
            # 对于本应用，sum(exact_order / unit)不会精确等于最小订货件数mini的主要原因是，输入的原始数据中"周转天数"是向下取整的，
            # 在计算过程中被省去的小数部分对后续结果的影响就会逐步放大；所以采用重新计算的准确周转天数参与运算，而不是原始数据中向下取整的周转天数。
            # 将靠近ceil且较大的值向上取整，将靠近floor且较小的值向下取整，比全部取floor更利于实际销售。
            final_unit = pd.concat([ceil_unit[(ceil_unit - exact_unit) / ceil_unit <= 0.01],
                                    floor_unit[(ceil_unit - exact_unit) / ceil_unit > 0.01]]).sort_index()
            total_unit = sum(final_unit)
            if total_unit < mini:
                final_unit[final_unit[final_unit.values == final_unit.max()].index.values[0]] = \
                    final_unit[final_unit.values == final_unit.max()] + (mini - total_unit)
                total_unit = sum(final_unit)
            else:
                final_unit[final_unit[final_unit.values == final_unit.max()].index.values[0]] = \
                    final_unit[final_unit.values == final_unit.max()] - (total_unit - mini)
                total_unit = sum(final_unit)

            trunc_order = final_unit * unit
            for _ in trunc_order.index:
                order_ori[order_ori.index == _] = trunc_order[trunc_order.index == _]
                alr_uni[alr_uni.index == _] = final_unit[final_unit.index == _]
                # 因为一维数组trunc_order和一维数组order_ori长度不同，会报警"SettingWithCopyWarning:
                # A value is trying to be set on a copy of a slice from a DataFrame"，但不影响计算。
            print('最优平衡周转天数的精确值:', X, '\n', '未补货单品中，周转天数小于最优平衡周转天数的个数:', len(T_u[T_u < X]), '\n',
                  '进行补货量追加的单品，所需增量的精确值:', '\n', order_delta, '\n', '进行补货量追加的单品的最终补货件数:',  '\n', final_unit, '\n',
                  '该供应商下所有单品的最终补货量：', '\n', order_ori, '\n', '该供应商下所有单品的最终补货件数：', '\n', alr_uni, '\n',
                  '总订货件数:', total_unit, '\n')
            T_final = (df['stock'] + df['arriving'] + order_ori) / df['dms']
            for _ in order_ori.index:
                df_all['final-ordering'][df_all['final-ordering'].index == _] = order_ori[order_ori.index == _]
                df_all['final-unit'][df_all['final-unit'].index == _] = alr_uni[alr_uni.index == _]
                df_all['final-ret-days'][df_all['final-ret-days'].index == _] = round(T_final[T_final.index == _], 3)

        else:
            mini = pd.Series(list(set(df['mini-order-unit'])))[0]
            unit = np.longdouble(df['unit'])
            stock, arri, order, dms = df['stock'], df['arriving'], df['ordering'], df['dms']
            T = (stock + arri + order) / dms
            alr_uni = np.array(order / unit)

            X = ((mini - alr_uni) * unit + stock + arri + order) / dms
            order_delta = (X - T) * dms
            exact_order = order + order_delta

            exact_unit = exact_order / unit
            ceil_unit, floor_unit = np.ceil(exact_unit), np.floor(exact_unit)
            if ((ceil_unit - exact_unit) / ceil_unit).values <= 0.01:
                final_unit = ceil_unit
            else:
                final_unit = floor_unit
            final_order = final_unit * unit
            print('最优平衡周转天数的精确值:', X, '\n', '该单品所需补货增量的精确值:', '\n', order_delta, '\n',
                  '该单品的最终补货量（即总订货量）：', '\n', final_order, '\n', '该单品的最终补货件数:',  '\n', final_unit, '\n')
            T_final = (df['stock'] + df['arriving'] + final_order) / df['dms']
            df_all['final-ordering'][df_all['final-ordering'].index == final_order.index.values[0]] = final_order[final_order.index == final_order.index.values[0]]
            df_all['final-unit'][df_all['final-unit'].index == final_unit.index.values[0]] = final_unit[final_unit.index == final_unit.index.values[0]]
            df_all['final-ret-days'][df_all['final-ret-days'].index == T_final.index.values[0]] = round(T_final[T_final.index == T_final.index.values[0]], 3)

    return df_all


df_all_ori = pd.read_excel('/Users/zc/Desktop/常规V2.0+deepar/order_balance.xlsx')
df_all = df_all_ori.copy()
print('所有供应商的编码：', '\n', set(df_all['provider']), '\n', '供应商总个数：', len(set(df_all['provider'])), '\n',
      '数据维度：', '\n', list(df_all.columns), '\n')
df_all = opt_baln_order(df_all)
df_all.to_excel('/Users/zc/Desktop/常规V2.0+deepar/order_balance_final.xlsx')
