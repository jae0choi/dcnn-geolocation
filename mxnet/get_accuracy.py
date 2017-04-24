import s2
import argparse

def latlng_to_cell(lat, lng):
    latlng = s2.S2LatLng.FromDegrees(lat, lng)
    return s2.S2CellId.FromLatLng(latlng)

def get_cellid_from_token(token):
    return s2.S2CellId.FromToken(token)

def latlng_in_grid(lat, lng, token):
    latlng_cellid = latlng_to_cell(lat, lng)
    grid_cellid = get_cellid_from_token(token)
    return grid_cellid.contains(latlng_cellid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pred", help="Predictions with GPS coords and grids.")
    parser.add_argument("grids", help="A list of grids.")
    parser.add_argument("lst", help=".lst file for going from index to latlng")
    parser.add_argument("latlng", help="Test data with GPS coords and grid labels.")
    args = parser.parse_args()

    predictions = dict()
    with open(args.pred, 'r') as f:
        for line in f:
            data = line.strip().split('\t')
            idx = data[0]
            predictions[idx] = [int(data[i]) for i in range(1,10,2)]

    grid_tokens = dict()
    with open(args.grids, 'r') as f:
        i = 0
        for line in f:
            data = line.strip().split('\t')
            token = data[0]
            grid_tokens[i] = token
            i += 1
        print i

    id2hash = dict()
    with open(args.lst, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            h = line[2].split('/')[-1][:-4]
            id2hash[line[0]] = h

    latlng_labels = dict()
    with open(args.latlng, 'r') as f:
        for line in f:
            data = line.strip().split('\t')
            lat = float(data[3])
            lng = float(data[2])
            h = data[-1]
            latlng_labels[h] = (lat, lng)

    num_correct = 0
    num_correct_top5 = 0
    num_total = 0
    for idx, grids in predictions.iteritems():
        h = id2hash[idx]
        real_lat, real_lng = latlng_labels[h]
        correct = [int(latlng_in_grid(real_lat, real_lng, grid_tokens[g])) for g in grids]
        num_correct += correct[0]
        num_correct_top5 += max(correct)
        num_total += 1
    acc = float(num_correct) / num_total * 100
    acc_top5 = float(num_correct_top5) / num_total * 100
    print ("Num Correct:", num_correct, "Num Total:", num_total, "Acc:", acc, "Acc_Top5:", acc_top5)
