import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def opt_baln_order(df_all):
    """
    周转天数和已定件数是非独立变量，周转天数由（库存+在途+计算订货量）/日均销量（向下取整）得到，已定件数由计算订货量/商品件规格（须为整数）得到。
    其中因为周转天数向下取整，在计算过程中会产生传播误差，所以公式中不应使用取整后的周转天数，而应使用准确的小数型周转天数。
    """

    if df_all.isna().any().any():
        raise Exception('原数据中含有空值，请检查')
    df_all = pd.concat([df_all, pd.DataFrame(columns=list('ABC'))]).fillna(0)
    df_all.rename(columns={"A": "最终订货量", "B": "最终订货件数", 'C': '最终周转天数'}, inplace=True)
    for i in range(len(set(df_all['供应商']))):
        df_ori = df_all[df_all['供应商'] == list(set(df_all['供应商']))[i]]
        df = df_ori.copy()  # 使用copy方法另存一份df，否则df会和df_ori使用同一个存储地址
        # 将df中日均销量<=0的行剔除，以保证后续计算周转天数的准确性。当日均销量为0时，原数据中对应周转天数被置为0，表示极度缺货，而其实可能并不是。
        df['日均销量'] = df['日均销量'][df['日均销量'] > 0]
        df.dropna(inplace=True)
        print('第%s个供应商编码：' % (i+1), list(set(df['供应商']))[-1], '\n', '该供应商下日均销量大于0的单品数：', len(df), '\n')

        if len(df) > 1:

            order_ori, T_ori, unit_ori = df['计算订货量'], df['周转天数'], df['商品件规格（个）']
            T = (df['当前库存'] + df['在途库存'] + order_ori) / df['日均销量']  # 第一次未截断前，商品的精确周转天数
            if sum(T - T_ori >= 1) > 0:
                raise Exception('原始数据中周转天数向下取整时有误，请检查')
            critic = T.max()

            T_u = (df[T > critic]['当前库存'] + df[T > critic]['在途库存'] + order_ori[T > critic]) / df[T > critic]['日均销量']
            df_trunc = df[T <= critic]  # 若因df['日均销量']中存在0，而使T中存在nan，则T <= critic会被判定为false，则会被排除掉而不进入df_trunc
            mini = pd.Series(list(set(df_trunc['供应商最小起订量（件）'])))[0]  # set取集合，即只取样本中的不同元素；再将元素作为值取出，方便后续计算
            unit = np.longfloat(df_trunc['商品件规格（个）'])  # 商品件规格应转为计算机的最高存储和计算位数来保存和计算，
            # 否则当一个供应商下需分配的单品过多时，prod()无法算出连乘项，则会返回0；而使用循环会报"overflow error"也无法算出。
            stock, arri, order, dms = df_trunc['当前库存'], df_trunc['在途库存'], df_trunc['计算订货量'], df_trunc['日均销量']
            T_d = (stock + arri + order) / dms
            alr_uni = order_ori / unit_ori  # 未被截断前各单品的已定件数

            A = stock + arri + order  # 数量, series

            B = []
            for _ in range(len(unit)):
                unit_m = np.ma.array(unit, mask=False)
                unit_m.mask[_] = True
                B.append(unit_m.prod())
            B = np.array(B)  # array of longfloat

            pai_unit = unit.prod()  # prod()对序列做连乘
            # 循环方式对序列做连乘
            # pai_unit = 1
            # for i in range(len(unit)):
            #     pai_unit = pai_unit*unit[i]
            if not (sum(pai_unit / B - unit) < 1e-10):
                raise Exception('原始数据中商品件规格有误，导致连乘项B[i]计算有误；或截断后单品个数小于2')

            C = (mini - sum(alr_uni)) * pai_unit  # longfloat
            if C <= 0:
                print('供应商%s，已定件数之和大于等于供应商最小起订件数，无需追加补货' % list(set(df['供应商']))[-1])
                continue
            # return df['计算订货量']

            X = (C + np.dot(np.array(A), B)) / np.dot(np.array(dms), B)  # X是最优平衡周转天数的精确值，longfloat
            order_delta = (X - T_d) * dms  # 各单品所需增加补货的数量的精确值，series of longfloat

            j, k = 0, 0
            while sum(order_delta < 0) > 0:
                j += 1
                print('补货量增量不能为负，应当降低critic取值；这是第%s次向下二分搜索最优平衡周转天数' % j)
                critic = critic / 2  # 用二分法向下寻找最优临界值
                T_u = (df[T > critic]['当前库存'] + df[T > critic]['在途库存'] + order_ori[T > critic]) / df[T > critic]['日均销量']
                df_trunc = df[T <= critic]
                unit = np.longfloat(df_trunc['商品件规格（个）'])
                stock, arri, order, dms = df_trunc['当前库存'], df_trunc['在途库存'], df_trunc['计算订货量'], df_trunc['日均销量']
                T_d = (stock + arri + order) / dms

                A = stock + arri + order  # 数量, series

                B = []
                for _ in range(len(unit)):
                    unit_m = np.ma.array(unit, mask=False)
                    unit_m.mask[_] = True
                    B.append(unit_m.prod())
                B = np.array(B)  # array of longfloat

                pai_unit = unit.prod()
                if not (sum(pai_unit / B - unit) < 1e-10):
                    raise Exception('连乘项B[i]计算有误，或截断后单品个数小于2')

                C = (mini - sum(alr_uni)) * pai_unit  # longfloat

                X = (C + np.dot(np.array(A), B)) / np.dot(np.array(dms), B)  # X是最优平衡周转天数的精确值，longfloat
                order_delta = (X - T_d) * dms  # 各单品所需增加补货的数量的精确值，series of longfloat

                while len(T_u[T_u < X]) > 0:
                    k += 1
                    print('critic取值过小，筛选掉过多单品，使最优平衡周转天数过大，应增加critic取值；这是第%s次向上二分搜索最优平衡周转天数' % k)
                    critic = critic * 1.5  # 用二分法向上寻找最优临界值
                    T_u = (df[T > critic]['当前库存'] + df[T > critic]['在途库存'] + order_ori[T > critic]) / df[T > critic]['日均销量']
                    df_trunc = df[T <= critic]
                    unit = np.longfloat(df_trunc['商品件规格（个）'])
                    stock, arri, order, dms = df_trunc['当前库存'], df_trunc['在途库存'], df_trunc['计算订货量'], df_trunc['日均销量']
                    T_d = (stock + arri + order) / dms

                    A = stock + arri + order  # 数量, series

                    B = []
                    for _ in range(len(unit)):
                        unit_m = np.ma.array(unit, mask=False)
                        unit_m.mask[_] = True
                        B.append(unit_m.prod())
                    B = np.array(B)  # array of longfloat

                    pai_unit = unit.prod()
                    if not (sum(pai_unit / B - unit) < 1e-10):
                        raise Exception('连乘项B[i]计算有误，或截断后单品个数小于2')

                    C = (mini - sum(alr_uni)) * pai_unit  # longfloat

                    X = (C + np.dot(np.array(A), B)) / np.dot(np.array(dms), B)
                    order_delta = (X - T_d) * dms
            print('总共搜索%s次找到最优平衡周转天数' % (j+k), '\n')

            exact_order = order + order_delta  # 各单品总补货量的精确值，longfloat
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
            T_final = (df['当前库存'] + df['在途库存'] + order_ori) / df['日均销量']
            for _ in order_ori.index:
                df_all['最终订货量'][df_all['最终订货量'].index == _] = order_ori[order_ori.index == _]
                df_all['最终订货件数'][df_all['最终订货件数'].index == _] = alr_uni[alr_uni.index == _]
                df_all['最终周转天数'][df_all['最终周转天数'].index == _] = T_final[T_final.index == _]

        else:
            mini = pd.Series(list(set(df['供应商最小起订量（件）'])))[0]
            unit = np.longfloat(df['商品件规格（个）'])
            stock, arri, order, dms = df['当前库存'], df['在途库存'], df['计算订货量'], df['日均销量']
            T = (stock + arri + order) / dms
            alr_uni = np.array(order / unit)

            if mini - alr_uni <= 0:
                print('已定件数之和大于等于供应商最小起订件数，无需追加补货')
                continue
                # return df['计算订货量']

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
            T_final = (df['当前库存'] + df['在途库存'] + final_order) / df['日均销量']
            df_all['最终订货量'][df_all['最终订货量'].index == final_order.index.values[0]] = final_order[final_order.index == final_order.index.values[0]]
            df_all['最终订货件数'][df_all['最终订货件数'].index == final_unit.index.values[0]] = final_unit[final_unit.index == final_unit.index.values[0]]
            df_all['最终周转天数'][df_all['最终周转天数'].index == T_final.index.values[0]] = T_final[T_final.index == T_final.index.values[0]]

    return df_all


df_all_ori = pd.read_excel('C:/Users/admin/Desktop/order_balance.xlsx')
df_all = df_all_ori.copy()  # 使df_all_ori和df_all的存储地址分离
print('所有供应商的编码：', '\n', set(df_all['供应商']), '\n', '供应商总个数：', len(set(df_all['供应商'])), '\n',
      '数据维度：', '\n', list(df_all.columns), '\n')
opt_baln_order(df_all)
