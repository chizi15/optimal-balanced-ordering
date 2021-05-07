import pandas as pd
import numpy as np
import warnings
pd.set_option('mode.chained_assignment', None)


def opt_baln_order(df_all):
    """
    本函数实现的功能：
    找到各个供应商下所有单品在当前状态下的最优平衡周转天数，低于该天数的商品一律通过追加补货量的方式拉齐到该平衡周转天数，高于该天数的商品则不变；
    且使各个单品的总补货量为自身件规格的整数倍；且使供应商下所有单品的最终补货件数之和恰好等于该供应商的最小起订件数；
    且尽可能减少最优平衡周转天数的搜索次数，使计算速度加快。
    :param：传入一个或多个供应商的补货相关信息的df，包括的数据维度有['code', 'provider', 'mini-order-unit',
    'unit', 'stock', 'arriving', 'ordering', 'dms', 'retention-days', 'already-unit']。
    :return：返回一个或多个供应商的最终补货信息的df，在传入的df_all的基础上，增加了['final-ordering', 'final-unit',
    'final-ret-days', 'ret-deviation(%)']四个维度。
    """

    if df_all.isna().any().any():
        raise Exception('原数据中含有空值，可能是‘retention-days’列在某些行为空，因为此时该行对应dms为0，使周转天数为无穷大')
    df_all = pd.concat([df_all, pd.DataFrame(columns=list('ABCD'))]).fillna(0)
    df_all.rename(columns={"A": "final-ordering", "B": "final-unit", 'C': 'final-ret-days', 'D': 'ret-deviation(%)'},
                  inplace=True)
    for i in range(len(set(df_all['provider']))):
        df_ori = df_all[df_all['provider'] == list(set(df_all['provider']))[i]]
        df = df_ori.copy()  # 使用copy方法另存一份df，否则df会和df_ori使用同一个存储地址，则更改会同时起作用
        if sum(df['already-unit']) >= pd.Series(list(set(df['mini-order-unit'])))[0]:
            for _ in df['already-unit'].index:
                df_all['final-ordering'][df_all['final-ordering'].index == _] = df['ordering'][
                    df['ordering'].index == _]
                df_all['final-unit'][df_all['final-unit'].index == _] = df['already-unit'][
                    df['already-unit'].index == _]
                # 使展示的最终周转天数的小数位数与原始数据中日均销售的小数位数相同，均为小数点后3为
                df_all['final-ret-days'][df_all['final-ret-days'].index == _] = \
                    round((df['stock'] + df['arriving'] + df['ordering']) / df['dms'], 3)
            print('provider %s 下的单品无需追加补货，因其已定件数总量不小于最小起订量' % int(list(set(df['provider']))[-1]))
            continue  # 跳过后续所有语句，进行下一次"for i in range(len(set(df_all['provider'])))"循环

        # 检查各个维度中数据的类型是否正确
        if sum(np.ceil(df['mini-order-unit']) != df['mini-order-unit']) > 0:
            print('\n', '供应商 %s 的最小起订件数中存在非整数' % int(list(set(df['provider']))[-1]))
            warnings.warn('最小起订件数中存在非整数', category=UserWarning)
        if len(set(df['mini-order-unit'])) != 1:
            print('\n', '供应商 %s 的最小起订件数不唯一' % int(list(set(df['provider']))[-1]))
            warnings.warn('供应商最小起订件数不唯一', category=UserWarning)
        if sum(np.ceil(df['unit']) != df['unit']) > 0:
            print('\n', '供应商 %s 的件规格中存在非整数' % int(list(set(df['provider']))[-1]))
            warnings.warn('件规格列存在非整数', category=UserWarning)
        if sum(np.ceil(df['stock']) != df['stock']) > 0:
            print('\n', '供应商 %s 的库存中存在非整数' % int(list(set(df['provider']))[-1]))
            warnings.warn('库存列存在非整数', category=UserWarning)
        if sum(np.ceil(df['arriving']) != df['arriving']) > 0:
            print('\n', '供应商 %s 的在途中存在非整数' % int(list(set(df['provider']))[-1]))
            warnings.warn('在途列存在非整数', category=UserWarning)
        if sum(np.ceil(df['ordering']) != df['ordering']) > 0:
            print('\n', '供应商 %s 的初始订货量中存在非整数' % int(list(set(df['provider']))[-1]))
            warnings.warn('初始订货量中存在非整数', category=UserWarning)

        # 将df中日均销量<=0的行剔除，以保证后续计算周转天数的准确性。当日均销量为0时，会使计算周转天数为无穷大，此时原数据中对应周转天数被置为0，
        # 表示极度缺货，而实际可能是很长时间没有销售；对于这种情况，则不进行追加补货，以防很久都卖不出去。
        df['dms'] = df['dms'][df['dms'] > 0]
        df.dropna(inplace=True)
        print('\n', '第%s个供应商编码：' % (i + 1), int(list(set(df['provider']))[-1]), '\n', '该供应商下日均销量大于0的单品数：', len(df),
              '\n')

        if len(df) > 1:  # 当有可能参与追加补货的有效单品数的最大值>=2时
            # 周转天数和已定件数是非独立变量，周转天数由（库存 + 在途 + ordering） / dms（向下取整）得到，
            # 已定件数由计算订货量 / 商品件规格（须为整数）得到。其中因为周转天数向下取整，在计算过程中会产生传播误差，
            # 所以公式中不应使用取整后的周转天数，而应使用准确的小数形式周转天数。
            order_ori, T_ori, unit_ori = df['ordering'], df['retention-days'], df['unit']
            T = (df['stock'] + df['arriving'] + order_ori) / df['dms']  # 第一次未截断前，各商品的精确周转天数
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

            j, k, m = 0, 0, 0
            while sum(order_delta < 0) > 0:
                j += 1
                print('补货量增量不能为负，应当降低critic取值；这是第 %s 次向下二分搜索最优平衡周转天数' % j)
                critic = critic / 2  # 用二分法向下搜索最优临界值
                T_u = (df[T > critic]['stock'] + df[T > critic]['arriving'] + order_ori[T > critic]) / df[T > critic][
                    'dms']
                df_trunc = df[T <= critic]
                unit = np.longdouble(df_trunc['unit'])
                stock, arri, order, dms = df_trunc['stock'], df_trunc['arriving'], df_trunc['ordering'], df_trunc['dms']
                T_d = (stock + arri + order) / dms

                if len(df_trunc) < 2:  # 上下二分搜索已不能找到最优平衡周转天数，改用顺序搜索重头开始
                    # T_des = T.sort_values(ascending=False, ignore_index=True)  # 高版本pandas才支持ignore_index参数
                    T_des = pd.Series(T.sort_values(ascending=False).values)
                    for i in range(len(T_des)):
                        m += 1
                        print('上下二分搜索终止，不能找到最优周转天数，改为顺序重头搜索；这是第 %s 次顺序搜索最优平衡周转天数' % m)
                        critic = T_des[i]
                        T_u = (df[T > critic]['stock'] + df[T > critic]['arriving'] + order_ori[T > critic]) / \
                              df[T > critic][
                                  'dms']
                        df_trunc = df[T <= critic]
                        unit = np.longdouble(df_trunc['unit'])
                        stock, arri, order, dms = df_trunc['stock'], df_trunc['arriving'], df_trunc['ordering'], \
                                                  df_trunc[
                                                      'dms']
                        T_d = (stock + arri + order) / dms

                        if len(df_trunc) == 1:
                            print('顺序搜索时，截断后只剩一个单品，改用单一商品算法')
                            mini = pd.Series(list(set(df_trunc['mini-order-unit'])))[0]
                            unit = np.longdouble(df_trunc['unit'])
                            stock, arri, order, dms = df_trunc['stock'], df_trunc['arriving'], df_trunc['ordering'], \
                                                      df_trunc['dms']
                            T = (stock + arri + order) / dms
                            X = (((mini - sum(alr_uni)) * unit + stock + arri + order) / dms).values[0]
                            order_delta = (X - T) * dms
                            break  # 退出“for i in range(len(T_des))”

                        else:
                            A = stock + arri + order
                            B = []
                            for _ in range(len(unit)):
                                unit_m = np.ma.array(unit, mask=False)
                                unit_m.mask[_] = True
                                B.append(unit_m.prod())
                            B = np.array(B)  # array of longdouble
                            pai_unit = unit.prod()
                            C = (mini - sum(alr_uni)) * pai_unit  # longdouble
                            X = (C + np.dot(np.array(A), B)) / np.dot(np.array(dms), B)
                            order_delta = (X - T_d) * dms
                            if (sum(order_delta >= 0) == len(order_delta)) and (len(T_u[T_u > X]) == len(T_u)) and (
                                    len(T_u) != 0):
                                break  # 退出当前“for i in range(len(T_des))”
                    break  # 退出外层“while sum(order_delta < 0) > 0”

                else:
                    A = stock + arri + order  # 数量, series
                    B = []
                    for _ in range(len(unit)):
                        unit_m = np.ma.array(unit, mask=False)
                        unit_m.mask[_] = True
                        B.append(unit_m.prod())
                    B = np.array(B)  # array of longdouble
                    pai_unit = unit.prod()
                    if not (sum(pai_unit / B - unit) < 1e-10):
                        raise Exception('向下二分搜索时连乘项B[i]计算有误，可能是unit数据类型的精度不够')
                    C = (mini - sum(alr_uni)) * pai_unit  # longdouble
                    X = (C + np.dot(np.array(A), B)) / np.dot(np.array(dms), B)  # X是最优平衡周转天数的精确值，longdouble

                order_delta = (X - T_d) * dms  # 各单品所需增加补货的数量的精确值，series of longdouble
                # 由于df_trunc = df[T <= critic]，(X - T_d)可以保证始终为正，及始终只对小于平衡周转天数的那些单品追加补货，
                # 而不会对大于平衡周转天数的那些单品削减补货。

                while len(T_u[T_u <= X]) > 0:
                    k += 1
                    print('critic取值过小，筛选掉过多单品，使最优平衡周转天数过大，应增加critic取值；这是第%s次向上二分搜索最优平衡周转天数' % k)
                    critic = critic * 1.5  # 用二分法向上搜索最优临界值
                    T_u = (df[T > critic]['stock'] + df[T > critic]['arriving'] + order_ori[T > critic]) / \
                          df[T > critic]['dms']
                    df_trunc = df[T <= critic]
                    unit = np.longdouble(df_trunc['unit'])
                    stock, arri, order, dms = df_trunc['stock'], df_trunc['arriving'], df_trunc['ordering'], df_trunc[
                        'dms']
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
                        raise Exception('向上二分搜索时连乘项B[i]计算有误，可能是unit数据类型的精度不够')
                    C = (mini - sum(alr_uni)) * pai_unit  # longdouble
                    X = (C + np.dot(np.array(A), B)) / np.dot(np.array(dms), B)

                    order_delta = (X - T_d) * dms

            print('总共搜索 %s 次找到最优平衡周转天数' % (j + k + m), '\n')

            if (sum(np.ceil(alr_uni) != alr_uni) > 0) or (sum(df['already-unit'] - alr_uni) != 0):
                print('\n', '供应商 %s 中存在初始订货量不是件规格的整数倍，或初始已定件数不是整数的情况' % int(list(set(df['provider']))[-1]), '\n')
                warnings.warn('初始订货量不是件规格的整数倍，或初始已定件数不是整数')
            # 理论上应当用np.ceil()，对各单品的最终补货件数，向上取整，为了满足供应商最小订货件数的要求；
            # 但实际上使用np.floor()更好，因为补货件数少的单品通常dms较低，如果向上取整多补零点几件虽然能满足供应商最小订货件数的要求，
            # 但会较大幅度地增加这种低销量单品的周转天数，对实际销售不利。所以采用np.floor()，如果最终订货件数不足，则去增加销量最大单品的订货件数，
            # 也不会使其增加多少周转天数。sum(exact_order / unit)不会精确等于最小订货件数mini，有时甚至有一定程度的偏差，是因为传播误差的存在。
            # 对于无理数或无限循环小数，当小数点前的位数越多，由于计算机存储位数的有限性，会截断小数点后越大的位数；
            # 当经过多次运算，特别是乘除法、开根号等，以及参与运算的数值越大，则会产生越大的传播误差。但通常情况下，传播误差很小，对实际结果基本不影响。
            # 对于本应用，sum(exact_order / unit)不会精确等于最小订货件数mini的主要原因是，输入的原始数据中"周转天数"是向下取整的，
            # 在计算过程中被省去的小数部分对后续结果的影响就会逐步放大；所以采用重新计算的准确周转天数参与运算，而不是原始数据中向下取整的周转天数。
            # 将靠近ceil且较大的值向上取整，将靠近floor且较小的值向下取整，比全部取floor更利于实际销售。
            if len(order_delta) == 1:  # 当有可能参与追加补货的有效单品数的最大值>=2，但实际需要追加补货的单品只有一个时
                exact_unit = order_delta / unit
                ceil_unit, floor_unit = np.ceil(exact_unit), np.floor(exact_unit)
                total_unit = exact_unit + sum(alr_uni)
                if total_unit.values[0] < mini:
                    final_unit = ceil_unit
                else:
                    final_unit = floor_unit
                trunc_order = final_unit * unit
                index_value = trunc_order.index.values[0]
                order_ori[order_ori.index == index_value] = trunc_order[trunc_order.index == index_value] + \
                                                            order_ori[order_ori.index == index_value]
                alr_uni[alr_uni.index == index_value] = final_unit[final_unit.index == index_value] + \
                                                        alr_uni[alr_uni.index == index_value]

                if sum(alr_uni) != mini:
                    raise Exception('供应商 %s 最终总订货件数 %s 不等于最小起订量 %s' % (
                        int(list(set(df['provider']))[-1]), sum(alr_uni), mini))
                print('最优平衡周转天数的精确值:', X, '\n', '未补货单品中，周转天数小于最优平衡周转天数的个数:', len(T_u[T_u < X]), '\n',
                      '进行补货量追加的单品，所需增量的精确值:', '\n', order_delta, '\n', '进行补货量追加的单品，追加的搜整订货件数:', '\n',
                      final_unit, '\n',
                      '该供应商下所有单品的最终补货量：', '\n', order_ori, '\n', '该供应商下所有单品的最终订货件数：', '\n', alr_uni, '\n',
                      '总订货件数:', sum(alr_uni), '\n')

                T_final = (df['stock'] + df['arriving'] + order_ori) / df['dms']
                dev_ratio = (T_final - X) / X * 100
                for _ in order_ori.index:
                    df_all['final-ordering'][df_all['final-ordering'].index == _] = order_ori[
                        order_ori.index == _]
                    df_all['final-unit'][df_all['final-unit'].index == _] = alr_uni[alr_uni.index == _]
                    df_all['final-ret-days'][df_all['final-ret-days'].index == _] = round(
                        T_final[T_final.index == _], 3)
                    df_all['ret-deviation(%)'][df_all['ret-deviation(%)'].index == _] = round(
                        dev_ratio[dev_ratio.index == _], 1)

            else:  # 当有可能参与追加补货的有效单品数的最大值>=2，且实际需要追加补货的单品数也>=2时；前者单品数几乎全都大于后者单品数
                exact_unit = order_delta / unit
                ceil_unit, floor_unit = np.ceil(exact_unit), np.floor(exact_unit)
                final_unit = pd.concat([ceil_unit[(ceil_unit - exact_unit) / ceil_unit <= 0.01],
                                        floor_unit[(ceil_unit - exact_unit) / ceil_unit > 0.01]]).sort_index()
                total_unit = sum(final_unit) + sum(alr_uni)

                # 偏差量补齐
                if total_unit < mini:
                    # 若final_unit中存在多个最大值，则将所需增量(mini - total_unit)平分增加到每一个最大值索引处
                    final_unit_max = final_unit[final_unit.values == final_unit.max()]
                    final_unit_max = final_unit_max + (mini - total_unit) / len(final_unit_max)
                    for _ in final_unit_max.index:
                        final_unit[final_unit.index == _] = final_unit_max[final_unit_max.index == _]
                else:
                    # 若final_unit中存在多个最大值，则让每一个最大值索引处都减少((total_unit - mini)/len(final_unit_max))
                    final_unit_max = final_unit[final_unit.values == final_unit.max()]
                    final_unit_max = final_unit_max - (total_unit - mini) / len(final_unit_max)
                    for _ in final_unit_max.index:
                        final_unit[final_unit.index == _] = final_unit_max[final_unit_max.index == _]
                trunc_order = final_unit * unit

                # 增加量与原始量相加合并
                for _ in trunc_order.index:
                    # 需要进行追加补货的各单品的追加补货量trunc_order，加所有单品的初始补货量order_ori，等于所有单品的最终补货量order_ori
                    order_ori[order_ori.index == _] = trunc_order[trunc_order.index == _] + order_ori[
                        order_ori.index == _]
                    # 需要进行追加订货的各单品的追加订货件数final_unit，加所有单品的初始订货件数alr_uni，等于所有单品的最终订货件数alr_uni
                    alr_uni[alr_uni.index == _] = final_unit[final_unit.index == _] + alr_uni[alr_uni.index == _]
                    # 因为一维数组trunc_order和一维数组order_ori长度不同，会报警"SettingWithCopyWarning:
                    # A value is trying to be set on a copy of a slice from a DataFrame"，但不影响计算。

                # 若存在非整数的订货量，则对其整数化
                T_final = (df['stock'] + df['arriving'] + order_ori) / df['dms']
                # 若最终订货件数alr_uni中含有非整数，将周转天数最小的订货件数向上取整，其余向下取整
                if sum(np.ceil(alr_uni) != alr_uni) > 0:
                    alr_uni[T_final[T_final == T_final[np.ceil(alr_uni) != alr_uni].min()].index] = \
                        np.ceil(alr_uni[T_final[T_final == T_final[np.ceil(alr_uni) != alr_uni].min()].index])
                    # 上面已经将周转天数最小的订货件数向上取整，并赋值，则alr_uni的小数订货件数商品中已经不存在原周转天数最小的那个商品
                    for _ in T_final[np.ceil(alr_uni) != alr_uni].index:
                        alr_uni[alr_uni.index == _] = np.floor(alr_uni[alr_uni.index == _])
                    order_ori = alr_uni * unit_ori

                    # 参与整数化的单品，若操作之后存在订货件数相同的情况，则可能是它们被近似处理了小数点，
                    # 此时总订货件数可能不等于最小起订量，则需对其执行进一步的（循环）偏差量分配。
                    # 若总订货件数不足，则对真实周转天数依次当前最小的单品增加一个补货件数，直至相等；
                    # 若总订货件数超量，则对真实周转天数依次当前最大的单品减小一个补货件数，直至相等。
                    while sum(alr_uni) + 1 <= mini:
                        T_final = (df['stock'] + df['arriving'] + order_ori) / df['dms']
                        alr_uni[T_final == T_final[trunc_order.index].min()] += 1
                        order_ori = alr_uni * unit_ori
                    while sum(alr_uni) - 1 >= mini:
                        T_final = (df['stock'] + df['arriving'] + order_ori) / df['dms']
                        alr_uni[T_final == T_final[trunc_order.index].max()] -= 1
                        order_ori = alr_uni * unit_ori

                    if sum(np.ceil(alr_uni) != alr_uni) > 0:  # 再次检查分配后是否含有非整数的最终订货件数
                        raise Exception('最终订货件数alr_uni中含有非整数')

                    T_final = (df['stock'] + df['arriving'] + order_ori) / df['dms']  # 因为order_ori必定有更新，所以T_final一定要更新

                if sum(alr_uni) != mini:  # 再次检查分配后的最终订货件数是否等于最小起订量
                    raise Exception(
                        '供应商 %s 最终总订货件数 %s 不等于最小起订量 %s' % (int(list(set(df['provider']))[-1]), sum(alr_uni), mini))

                dev_ratio = (T_final - X) / X * 100
                for _ in order_ori.index:
                    df_all['final-ordering'][df_all['final-ordering'].index == _] = order_ori[order_ori.index == _]
                    df_all['final-unit'][df_all['final-unit'].index == _] = alr_uni[alr_uni.index == _]
                    df_all['final-ret-days'][df_all['final-ret-days'].index == _] = round(T_final[T_final.index == _],
                                                                                          3)
                    df_all['ret-deviation(%)'][df_all['ret-deviation(%)'].index == _] = round(
                        dev_ratio[dev_ratio.index == _], 1)

                print('最优平衡周转天数的精确值:', X, '\n', '未补货单品中，周转天数小于最优平衡周转天数的个数:', len(T_u[T_u < X]), '\n',
                      '进行补货量追加的单品，所需增量的精确值:', '\n', order_delta, '\n', '进行补货量追加的单品，追加的搜整订货件数:', '\n', final_unit, '\n',
                      '该供应商下所有单品的最终补货量：', '\n', order_ori, '\n', '该供应商下所有单品的最终订货件数：', '\n', alr_uni, '\n',
                      '总订货件数:', sum(alr_uni), '\n')

        else:  # 当有可能参与追加补货的有效单品数至多为1时
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

            if (np.ceil(alr_uni) != alr_uni)[0] or ((df['already-unit'] - alr_uni) != 0).values[0]:
                print('\n', '供应商 %s 中存在初始订货量不是件规格的整数倍，或初始已定件数不是整数的情况' % int(list(set(df['provider']))[-1]), '\n')
                warnings.warn('初始订货量不是件规格的整数倍，或初始已定件数不是整数')
            if exact_unit.values[0] > mini:
                final_unit = floor_unit
            else:
                final_unit = ceil_unit

            final_order = final_unit * unit
            if final_unit.values[0] != mini:
                raise Exception(
                    '供应商 %s 最终总订货件数 %s 不等于最小起订量 %s' % (int(list(set(df['provider']))[-1]), final_unit, mini))
            print('最优平衡周转天数的精确值:', X, '\n', '该单品所需补货增量的精确值:', '\n', order_delta, '\n',
                  '该单品的最终补货量（即总订货量）：', '\n', final_order, '\n', '该单品的最终补货件数:', '\n', final_unit, '\n')
            T_final = (df['stock'] + df['arriving'] + final_order) / df['dms']
            dev_ratio = (T_final - X) / X * 100
            df_all['final-ordering'][df_all['final-ordering'].index == final_order.index.values[0]] = final_order[
                final_order.index == final_order.index.values[0]]
            df_all['final-unit'][df_all['final-unit'].index == final_unit.index.values[0]] = final_unit[
                final_unit.index == final_unit.index.values[0]]
            df_all['final-ret-days'][df_all['final-ret-days'].index == T_final.index.values[0]] = round(
                T_final[T_final.index == T_final.index.values[0]], 3)
            df_all['ret-deviation(%)'][df_all['ret-deviation(%)'].index == dev_ratio.index.values[0]] = round(
                dev_ratio[dev_ratio.index == dev_ratio.index.values[0]], 1)
        print('供应商 %s 下所有单品平衡补货完毕' % int(list(set(df['provider']))[-1]), '\n')

    check = df_all['final-ordering'] / df_all['unit'] != df_all['final-unit']
    if sum(check) > 0:
        raise Exception('对于供应商 %s ，存在 \"最终订货量\" 除以 \"件规格\" 不等于 \"最终订货件数\" 的情况' % set(df_all['provider'][check].values))

    return df_all


df_all_ori = pd.read_excel('C:/Users/admin/Desktop/functions/order_balance_all.xlsx')
df_all = df_all_ori.copy()
print('所有供应商的编码：', '\n', set(df_all['provider']), '\n', '供应商总个数：', len(set(df_all['provider'])), '\n',
      '数据维度：', '\n', list(df_all.columns), '\n')
df_all = opt_baln_order(df_all)
df_all.to_excel('C:/Users/admin/Desktop/functions/order_balance_all_output.xlsx')
