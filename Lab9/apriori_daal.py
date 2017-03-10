from daal.data_management import FileDataSource, DataSourceIface
from daal.algorithms.association_rules import (
    Batch_Float64Apriori, data, largeItemsets, largeItemsetsSupport,
    antecedentItemsets, consequentItemsets, confidence, itemsetsSortedBySupport,
    rulesSortedByConfidence
)
import time

def printAprioriItemsetsToFile(large_itemsets_table, large_itemsets_support_table, filename):
    large_itemset_count = large_itemsets_support_table.getNumberOfRows()
    num_items_in_large_itemsets = large_itemsets_table.getNumberOfRows()

    large_itemsets = large_itemsets_table.getBlockOfRowsAsDouble(0, num_items_in_large_itemsets).flatten()
    large_itemsets_support_data = large_itemsets_support_table.getBlockOfRowsAsDouble(0, large_itemset_count).flatten()

    large_itemsets_array = [[] for x in range(large_itemset_count)]

    for i in range(num_items_in_large_itemsets):
        large_itemsets_array[int(large_itemsets[2 * i])].append(large_itemsets[2 * i + 1])

    support_array = [0] * large_itemset_count

    for i in range(large_itemset_count):
        support_array[int(large_itemsets_support_data[2 * i])] = large_itemsets_support_data[2 * i + 1]
    f = open(filename, 'w')
    f.write('Itemset;Support\n')

    for i in range(0, large_itemset_count):
        f.write('{')
        for l in range(len(large_itemsets_array[i]) - 1):
            f.write('{:.0f}, '.format(large_itemsets_array[i][l]))
        f.write('{:.0f}}};'.format(large_itemsets_array[i][len(large_itemsets_array[i]) - 1]))
        f.write('{:.0f}\n'.format(support_array[i]))
    f.close()

def printAprioriRulesToFile(left_items_table, right_items_table, confidence_table, filename):

    num_rules = confidence_table.getNumberOfRows()
    num_left_items = left_items_table.getNumberOfRows()
    num_right_items = right_items_table.getNumberOfRows()

    left_items = left_items_table.getBlockOfRowsAsDouble(0, num_left_items).flatten()
    right_items = right_items_table.getBlockOfRowsAsDouble(0, num_right_items).flatten()
    confidence = confidence_table.getBlockOfRowsAsDouble(0, num_rules).flatten()

    left_items_array = [[] for x in range(num_rules)]

    if num_rules == 0:
        print("\nNo association rules were found ")
        return

    for i in range(num_left_items):
        left_items_array[int(left_items[2 * i])].append(left_items[2 * i + 1])

    right_items_array = [[] for x in range(num_right_items)]

    for i in range(num_right_items):
        right_items_array[int(right_items[2 * i])].append(right_items[2 * i + 1])

    confidence_array = [0] * num_rules

    for i in range(num_rules):
        confidence_array[i] = confidence[i]

    f = open(filename, 'w')
    f.write('Rule;Confidence\n')

    for i in range(0, num_rules):
        f.write('{')
        for l in range(len(left_items_array[i]) - 1):
            f.write('{:.0f}, '.format(left_items_array[i][l]))
        f.write('{:.0f}}} -> {{'.format(left_items_array[i][len(left_items_array[i]) - 1]))

        for l in range(len(right_items_array[i]) - 1):
            f.write('{:.0f}, '.format(right_items_array[i][l]))
        f.write('{:.0f}}};'.format(right_items_array[i][len(right_items_array[i]) - 1]))

        f.write('{0:.3g}\n'.format(confidence_array[i]))
    f.close()



start = time.time()
for num in range(1):
    datasetFileName = 'daal_retail.csv'

    minSupport = 0.0015
    minConfidence = 0.8

    dataSource = FileDataSource(
        datasetFileName, DataSourceIface.doAllocateNumericTable, DataSourceIface.doDictionaryFromContext
    )
    dataSource.loadDataBlock()

    alg = Batch_Float64Apriori()
    alg.input.set(data, dataSource.getNumericTable())
    alg.parameter.minSupport = minSupport
    alg.parameter.minConfidence = minConfidence
    # alg.parameter.itemsetsOrder = itemsetsSortedBySupport
    # alg.parameter.rulesOrder = rulesSortedByConfidence

    res = alg.compute()
end = time.time()
print('Performance comparison. Time: %s seconds' % (end - start))

nt1 = res.get(largeItemsets)
nt2 = res.get(largeItemsetsSupport)
nt3 = res.get(antecedentItemsets)
nt4 = res.get(consequentItemsets)
nt5 = res.get(confidence)

printAprioriItemsetsToFile(nt1, nt2, 'Apriori_result.csv')
print('Number of rules %i' %nt5.getNumberOfRows())
printAprioriRulesToFile(nt3, nt4, nt5, 'Assosiation_rules.csv')