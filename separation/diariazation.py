import librosa
import numpy as np
import soundfile as sf
from classes import *
from setting import *

def trim_audio_data(audio_array, start_time, end_time, sr = 16000):
    return audio_array[int(start_time*sr):int(sr*end_time)+1]

def split_audios(audio, pipeline, separation_model, enh_model):
    num_speaker = int(audio.speakerNum)
    audio_arr, _ = sf.read(audio.filepath)
    diarization = pipeline(audio.filepath, num_speakers=num_speaker)
    tempfilename = "./tempaudio/"+audio.filepath.split('/')[-1].split('.')[0]+'_temp'


    diar_result = [Message(**{"startTime":turn.start, "endTime":turn.end, "speaker":str(int(speaker.split('_')[-1])), "seq":i, "mix": False}) for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True))]
    if diar_result[0].endTime > diar_result[1].startTime:
        diar_result[0].mix = True
    if diar_result[-1].startTime < diar_result[-2].endTime:
        diar_result[-1].mix = True
    for i in range(1,len(diar_result)-1):
        if (diar_result[i].startTime < diar_result[i-1].endTime) or (diar_result[i].endTime > diar_result[i+1].startTime):
            diar_result[i].mix = True

    start = 0
    end = 0
    flag = False
    mixed = []
    for ind, i in enumerate(diar_result):
        if i.mix:
            if not flag:
                start = i.startTime
                end = i.endTime
                flag = True
                mixed.append([[ind]])
            else:
                if i.startTime > end:
                    mixed[-1].append((start, end))
                    start = i.startTime
                    end = i.endTime
                    mixed.append([[ind]])
                else:
                    if end < i.endTime:
                        end = i.endTime
                    mixed[-1][0].append(ind)
        else:
            if flag:
                mixed[-1].append((start, end))
                flag = False
    if flag:
        mixed[-1].append((start,diar_result[-1].endTime))

    for t, mix in enumerate(mixed):
        cutted_audio = trim_audio_data(audio_arr, mix[-1][0], mix[-1][1])
        sf.write(tempfilename+str(i)+'.wav',cutted_audio,16000,format='WAV')
        est_sources = separation_model.separate_file(tempfilename+str(i)+'.wav')
        outputs = []
        for i in range(num_speaker):
            outputs.append(librosa.resample(np.array(est_sources[:, :, i].detach().cpu()), orig_sr=8000, target_sr=16000)[0])
        for ind, data in enumerate(mix[0]):
            start_time = diar_result[data].startTime - mix[-1][0]
            end_time = diar_result[data].endTime - mix[-1][0]
            diar_result[data].audio = trim_audio_data(outputs[ind%2], start_time, end_time)
    for i in diar_result:
        if i.mix == False:
            cutted_audio = trim_audio_data(audio_arr, i.startTime, i.endTime)
            if use_enh:
                sf.write(tempfilename+str(i)+'.wav',cutted_audio,16000,format='WAV')
                est_sources = enh_model.separate_file(path=tempfilename+str(i)+'.wav')
                cutted_audio = np.array(est_sources[:, :, 0].detach().cpu())
            i.audio = cutted_audio
    return diar_result