import numpy as np


def find_wave_smoothness(waves_list):
    """
    Estimate waves smoothness (average waves depth)

    Parameters:
    - waves_list : list of waves depth 
   
    """
    wave_difs = [abs(j-i) for i, j in zip(waves_list[:-1], waves_list[1:])]
    wave_smoothness = sum(wave_difs) / len(waves_list)
    return wave_smoothness


def find_layer(echodata, beam_dead_zone, in_a_row_thresh, layer_quantile, layer_strength_thresh, layer_size_thresh):
    """
    Find layer in the acoustic data in the echograms.

    Parameters:
    - echodata :acoustic data 
    - beam_dead_zone : size of echosounder near-zone (signal too strong at the top), number of bins to remove at the top of the echodata.
    - in_a_row_thresh : Number of signals in a row in one ping that has to be False (smaller than layer_strength_thresh) to stop the find_layer algorithm.
    - layer_quantile :Which quantile to use when looking at each row to find average echo strength.
    - layer_strength_thresh : Threshold to find layers at the top of the files. Signals stronger than threshhold will be considered part of layer.
    - layer_size_thresh : Threshold to define layers size at the top of the files. 
    """
    echodata[np.isnan(echodata)] = 0
    echodata = echodata[beam_dead_zone:]
    in_a_row = 0

    for n, row in enumerate(echodata):
        row = row[~np.isnan(row)]
        avg_val = np.quantile(row, layer_quantile)

        if avg_val < layer_strength_thresh:
            in_a_row += 1

        if in_a_row == in_a_row_thresh:
            break

    if n > layer_size_thresh:
        try: 
            layer = n + beam_dead_zone
            return layer
        
        except:
            return False
    else:
        return False


def find_waves(echodata, wave_thresh, in_a_row_waves, beam_dead_zone):
    """
    Find waves depth.

    Parameters:
    - echodata :acoustic data 
    - wave_thresh : Threshold to use when looking for waves. Echo needs to be stronger than this threshold to be classified as part of wave.
    - in_a_row_waves : Number of signals in a row that has to be false (<wave_thresh) to consider wave as finished.
    - beam_dead_zone : size of echosounder near-zone (signal too strong at the top), number of bins to remove at the top of the echodata.

    """
    echodata[np.isnan(echodata)] = 0

    line = []

    for i, ping in enumerate(echodata.T):

        in_a_row = 0
        found_limit = False

        # if depth[i] == depth[i]:
        #     ping_depth = int(depth[i])
        #     ping = ping[:ping_depth]

        for i, value in enumerate(ping):
            if value < wave_thresh:
                in_a_row += 1
            else:
                in_a_row = 0 
            if in_a_row == in_a_row_waves:
                found_limit = True 
                line.append(i-in_a_row)
                break
        if not found_limit:
            line.append(beam_dead_zone)


    for ping in range(echodata.shape[1]):
        echodata[:(line[ping]), ping] = 0

    wave_avg = sum(line) / len(line)
    wave_smoothness = find_wave_smoothness(line)
    
    return echodata, line, wave_avg, wave_smoothness
