# -*- coding: utf-8 -*-

import pandas as pd
import pdb
from tqdm import tqdm
import tiktoken

from openai import OpenAI
client = OpenAI(api_key="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")

df = pd.read_json("../Datasets/Hindi_summarization/XLSum/hindi_val.jsonl", lines=True)
# prompt = "\n\nAct as a summarization tool and create a hindi summary of the given hindi news article in 32-64 words and maximum 2 sentences. The length of the summary should be appropriate for the length and complexity of the original text, providing a clear and accurate overview without omitting any important information."

dict = {"idx": [], "gen_summ": []}

# def num_tokens_from_messages(messages, model="gpt-3.5-turbo-1106"):
#   """Returns the number of tokens used by a list of messages."""
#   try:
#       encoding = tiktoken.encoding_for_model(model)
#   except KeyError:
#       encoding = tiktoken.get_encoding("cl100k_base")
#   if model == "gpt-3.5-turbo-1106":  # note: future models may deviate from this
#       num_tokens = 0
#       for message in messages:
#           num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
#           for key, value in message.items():
#               num_tokens += len(encoding.encode(value))
#               if key == "name":  # if there's a name, the role is omitted
#                   num_tokens += -1  # role is always required and always 1 token
#       num_tokens += 2  # every reply is primed with <im_start>assistant
#       return num_tokens
#   else:
#       raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
#   See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-16k-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

# print(df['text'][0])

for idx in tqdm(df.index):
    model_name = "ft:gpt-3.5-turbo-1106:personal::8owZHC2k"
    text = df['text'][idx]
    # mod_text = f"'{text}' " + prompt
    mod_text = text
    mod_messages=[
        {
        "role": "system",
        "content": "You are a summarization assistant that provides good quality hindi summaries for hindi news articles"
        },
        {
        "role": "user",
        "content": mod_text
        }
    ]
    num_tok = num_tokens_from_messages(mod_messages)
    if (num_tok > 15000):
        diff = num_tok - 15000
        print("Shortening text on index: ", idx)
        encoding = tiktoken.encoding_for_model(model_name)
        text_enc = encoding.encode(text)
        text_enc = text_enc[:-diff]
        text = encoding.decode(text_enc)
        # mod_text = f"'{text}' " + prompt
        mod_text = text
        mod_messages=[
            {
            "role": "system",
            "content": "You are a summarization assistant that provides good quality hindi summaries for hindi news articles"
            },
            {
            "role": "user",
            "content": mod_text
            }
        ]

    response = client.chat.completions.create(
    model=model_name,
    # response_format={ "type": "json_object" },
    messages=mod_messages,
    temperature=1,
    max_tokens=1024,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    dict["idx"].append(idx)
    dict["gen_summ"].append(response.choices[0].message.content)

gen_df = pd.DataFrame(dict)
gen_df.to_csv("chatgpt_fine_tune_val_10.csv", index=False)

# response = client.chat.completions.create(
#   model="gpt-3.5-turbo-1106",
#   messages=[
#     {
#       "role": "system",
#       "content": "You are a helpful assistant that provides good quality hindi summaries for the article provided."
#     },
#     {
#       "role": "user",
#       "content": "'मेरे सहयोगी को ये बात एक महिला ने फोन पर कही. महिला फोन पर बुरी तरह रो रही थी. महिला के ब्वॉयफ्रेंड ने उन्हें धमकी दी थी कि वो उनकी नग्न तस्वीरें इंटरनेट पर डाल देगा. अगर उनके परिवार के लोगों ने ये तस्वीरें देख ली तो? या उनके ऑफिस के लोगों के सामने ये तस्वीरें आ गईं? ये सवाल उनके दिमाग में कौंध रहे थे. ये सोच-सोचकर उनके ज़हन में आत्महत्या तक का ख़्याल आया. साल 2015 में मैंने रिवेंज पॉर्न हेल्पलाइन शुरू की थी. इसके ज़रिए हम ऐसे लोगों की मदद करते हैं. इस सर्विस को चलाने के लिए हमें सरकार से आर्थिक मदद मिलती है. समाप्त जब कोई शख्स किसी की सहमति के बिना उसकी अतरंग तस्वीरें और वीडियो बांट देता है, तो हम इस हेल्पलाइन के ज़रिए पीड़ित की मदद करते हैं. 2015 में इंग्लैंड और वेल्स में इसे अपराध घोषित किया गया, जिसके तहत कम से कम दो साल की सज़ा का प्रावधान है. रिवेंज पॉर्न कोई नई चीज़ नहीं है. शुरुआत में जो भी मामले हमारे सामने आए वो काफी पुराने थे. एक महिला ने हमें बताया कि उनके एक्स-पार्टनर उनकी नग्न तस्वीरों और वीडियो को कई ब्लॉग, सोशल मीडिया और वेबसाइट्स पर डाल चुके हैं. वो बीते सात सालों से इस सामग्री को इंटरनेट से हटवाने की कोशिश कर रही हैं, लेकिन उन्हें कामयाबी नहीं मिल सकी. उन्होंने पुलिस में भी शिकायत की, लेकिन पुलिस भी उनकी कोई मदद नहीं कर पाई. वो बेहद निराश हो चुकी थीं. हमारे पास कई ऐसे मामले आते हैं जिनके पीछे महिलाओं के एक्स-पार्टनर होते हैं. ये मामले दो तरह के होते हैं: दक्षिण कोरिया स्पाई कैमरा पोर्न की चपेट में जब कोई हमसे संपर्क करता है तो हम सबसे पहले उसकी तस्वीरों को इंटरनेट से हटवाने की कोशिश करते हैं. हम इन्हें हटवाने की गारंटी नहीं दे सकते, क्योंकि कई ऐसी वेबसाइट्स होती हैं जो हमारा सहयोग नहीं करती और हमें पूरी तरह से नज़रअंदाज़ कर देती हैं. हेल्पलाइन शुरू करने के पहले साल हमें 3,000 फोन कॉल आए. तीन साल में हमें 12,000 से ज़्यादा कॉल और इमेल मिले. मैं ये नहीं कहूंगा कि पॉर्न रिवेंज के मामले बढ़े हैं, बल्कि लोग अब जागरुक हो रहे हैं और ऐसे मामलों में मदद मांग रहे हैं. \"सेल्फी जनरेशन\" पॉर्न रिवेंज का शिकार बनती है. आपने सुना भी होगा: लड़की ने बिना कपड़ों वाली फोटो ब्वॉयफ्रेंड को भेजी. इसके बाद दोनों का ब्रेक-अप हो गया, तो उसने लड़की की तस्वीरें इंटरनेट पर शेयर कर दीं. हमसे कई युवा मदद मांगते हैं. लेकिन हमारे सामने 40-50 साल की उम्र के लोगों के मामले भी आते हैं. सीक्रेट कैमरों की बदौलत, फल फूल रही है पोर्न इंडस्ट्री एक दिन हमसे 70 साल के एक शख्स ने संपर्क किया. पीड़ित को कोई ब्लैकमेल कर रहा था. किसी ने चुपके से उनके सेक्शुअल एक्ट की वीडियो बना ली और ब्लैकमेल करके पैसे मांगने लगा. हमें कॉल करने वाली ज़्यादातर महिलाएं होती हैं और एक चौथाई पुरुष. हाल ही में हमारे सामने एक मामला आया, जिसमें डेटिंग ऐप पर एक लड़की का फेक अकाउंट बनाकर किसी ने एक पुरुष से संपर्क किया. फ़ेक अकाउंट के उस पार बैठे व्यक्ति ने उन्हें उत्तेजित कर हस्तमैथुन करवाया. फिर इसी के ज़रिए ब्लैकमेल करके पैसे मांगने लगा. कई बार लोगों के नहाने या बेडरूम में निजी पलों की वीडियो बना ली जाती है. ऐसा करने वाले कई बार उनके जानने वाले ही होते हैं. या फिर कई बार लोगों के सोशल मीडिया अकाउंट्स हैक कर उनकी नग्न तस्वीरें चुरा ली जाती हैं. मुझे याद है मेरी एक क्लाइंट थीं, जो काफी लोकप्रिय हैं. उन्होंने सोशल मीडिया के ज़रिए किसी से अपनी अंतरंग तस्वीरें साझा की. लेकिन किसी ने उन तस्वीरों को निकालकर वायरल कर दिया. हमने उन तस्वीरों को इंटरनेट से हटाने की बहुत कोशिश की, लेकिन वो तस्वीरें हर जगह फैल चुकीं थीं. नहाती हुई 34 महिलाओं का वीडियो बनाने वाला दोषी क़रार लोगों को अपने फोन पर अंतरंग तस्वीरें रखने से बचना चाहिए. एक किशोरी के माता-पिता ने हमसे संपर्क किया था. उनकी बेटी का फ़ोन चोरी हो गया था. फ़ोन में उनकी बेटी की कुछ टॉपलेस तस्वीरें थी, जो उसने बीच पर लीं थी. ये तस्वीरें चोरों के हाथ लग गईं. चोरों ने लड़की को ब्लैकमेल करना शुरू कर दिया. यहां तक की उसके परिजनों को भी धमकी दी कि अगर उन्होंने पैसे नहीं दिए तो वो तस्वीरों को इंटरनेट पर डाल देंगे. हमने लड़की के माता-पिता को पुलिस में जाने की सलाह दी और कहा कि चोरों की कोई मांग ना मानें. कई बार अभियुक्त खुद हमसे संपर्क करते हैं. मुझे एक वाकया याद आता है. एक लड़के ने अपनी एक्स-गर्लफ्रेंड की तस्वीर ऑनलाइन पोस्ट कर दी, इसके बाद उसे अपनी इस हरकत पर पछतावा हुआ. जिस रिवेंज वेबसाइट पर उसने अपनी एक्स-गर्लफ्रेंड की तस्वीरें डाली थी, वो वेबसाइट खासकर इसलिए थी कि कोई अपने एक्स को शर्मिंदा करने के लिए नाम लिखकर तस्वीरें डाल सके. हमने उसकी मदद की और वो तस्वीरें वेबसाइट से हटवा दीं. हालांकि ये सब करने में कुछ वक्त लगा. हमने उस लड़के को बताया कि उसने क़ानून का उल्लंघन किया है. लेकिन जजमेंटल होने की बजाए हमारी प्राथमिकता लोगों की मदद करना है. अब ये उनकी पार्टनर पर है कि वो पुलिस को शिकायत करना चाहती है या नहीं. एक दफ़ा एक महिला के एक्स-पार्टनर ने उनकी नग्न तस्वीरें ऑफिस की कॉमन इमेल आईडी पर भेज दीं. ये मेल उनका हर सहकर्मी देख सकता था. हालांकि उनकी कंपनी के लोगों ने उनका काफ़ी सहयोग किया, लेकिन क्या आप सोच सकते हैं कि उस महिला पर क्या बीती होगी? पीड़िता कंपनी के एक ऊंचे पद पर थीं. उन्होंने ठान लिया कि वो अपने एक्स-पार्टनर को नहीं छोड़ेंगी. और उनकी ये सोच बिल्कुल सही थी. उन्होंने हमसे संपर्क किया और पुलिस को भी फोन मिला दिया. हमने उन्हें सलाह दी कि वो अपने सारे सहकर्मियों को वो मेल डिलीट करने को कहें. हमने उन्हें ये भी बताया कि उन्हें पुलिस को क्या सबूत देने हैं. लेकिन हम नहीं जानते कि आख़िर में उस केस का क्या हुआ? ज़्यादातर मामलों में हमें पता नहीं चलता कि आख़िर में क्या हुआ, क्योंकि हम अदालत की प्रक्रिया तक क्लाइंट के साथ नहीं रह पाते. 'ट्रंप ने पोर्न स्टार को दिए थे 1 लाख 30 हज़ार डॉलर' हमारी टीम पूरे उत्साह के साथ काम करती है. हमारा कोई बहुत बड़ा कॉल सेंटर नहीं है, हमारी टीम भी बहुत छोटी सी है. काम करने वाले हम सिर्फ तीन लोग हैं, इसलिए काम का काफी लोड होता है. रोज़ाना इस तरह के गंभीर मामलों से दो-चार होना हम पर भावनात्मक असर डालता है. कई मामले बहुत परेशान करने वाले होते हैं. हम सबकी मदद करना चाहते हैं, लेकिन ऐसा नहीं कर पाते. हमारे काम का एक और पहलु भी है. हमें बहुत-सा टाइम पॉर्न वेबसाइट पर बिताना पड़ता है. इन वेबसाइट्स का कंटेंट लगातार देखना परेशान कर देने वाला होता है. इस काम के लिए मुझे अपने आप को मज़बूत बनाना पड़ता है, क्योंकि हमें बहुत से लोगों की मदद करनी है. इस पेशे में मुझे कॉलरों से कुछ दूरी बनाकर रखनी पड़ती है, इसके बावजूद आप इमोशनल महसूस करने लगते हैं. उनकी कहानियां मेरे दिल को छूती हैं. मैं हमेशा अपने आप को याद दिलाता हूं कि मैं ये क्यों कर रहा हूं. जब काम बहुत ज़्यादा हो जाता है, तो मैं अपनी टीम से कहता हूं, \"एक ब्रेक लो. ये वेबसाइट्स बंद करो और कुछ समय के लिए कुछ और करो.\" किसी के पास भी ये अधिकार नहीं है कि वो किसी और की निजी तस्वीरों को यूं साझा कर दे. मैं उन लोगों के लिए लड़ना जारी रखूंगा. रिवेंज पोर्न हेल्पलाइन के संस्थापक से बातचीत पर आधारित. पहचान छिपाने के लिए इस लेख में कुछ जानकारी बदल दी गई है. पोर्न स्टार को धमकाने वाला क्या ट्रंप का आदमी था? (बीबीसी हिन्दी के एंड्रॉएड ऐप के लिए आप यहां क्लिक कर सकते हैं. आप हमें फ़ेसबुक, ट्विटर, इंस्टाग्राम और यूट्यूब पर फ़ॉलो भी कर सकते हैं.)' \n\nAct as a summarization tool and create a hindi summary of the given hindi news article in 32-64 words and maximum 2 sentences. The length of the summary should be appropriate for the length and complexity of the original text, providing a clear and accurate overview without omitting any important information."
#     }
#   ],
#   temperature=1,
#   max_tokens=1024,
#   top_p=1,
#   frequency_penalty=0,
#   presence_penalty=0
# )

# print(response)


# response = client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {
#       "role": "system",
#       "content": "You are a helpful assistant that provides good quality hindi summaries for the article provided."
#     },
#     {
#       "role": "user",
#       "content": "'ख़बरें हैं कि बिहार के आरा, भोजपुर, मुज़्ज़्फ़रपुर ज़िलों में सड़कों पर आगजनी और हिंसक झड़पें हुई हैं. बिहार से सीटू तिवारी की रिपोर्ट भारत बंद को लेकर हिंसा की आशंका को देखते हुए राजस्थान के जयपुर और मध्य प्रदेश के भोपाल में धारा 144 लगा दी गई है. राजस्थान के झालावाड़ ज़िले में बाज़ार बंद होने की रिपोर्ट है. आरक्षण का विरोध कर रहे लोगों ने बाइक रैली का आयोजन किया. भारत बंद के मद्देनज़र राजस्थान, मध्य प्रदेश और अन्य राज्यों में प्रशासन को सतर्क रहने के लिए पहले ही कह दिया गया था. समाचार एजेंसी पीटीआई के अनुसार गृह मंत्रालय ने कहा है कि किसी भी तरह की हिंसा के लिए ज़िले के डीएम और एसपी को ज़िम्मेदार माना जाएगा. इससे पहले एससी/एसटी क़ानून में छेड़छाड़ का आरोप लगाकर दलित और आदिवासी समाज के लोगों ने दो अप्रैल को भारत बंद कराया था. इस समूह का कहना था कि सुप्रीम कोर्ट के एक फ़ैसले से दलितों और आदिवासियों के होने वाले शोषण के ख़िलाफ़ क़ानून को कमज़ोर बनाया गया है. (बीबीसी हिन्दी के एंड्रॉएड ऐप के लिए आप यहां क्लिक कर सकते हैं. आप हमें फ़ेसबुक और ट्विटर पर फ़ॉलो भी कर सकते हैं.)' \n\nAct as a summarization tool and create a hindi summary of the given hindi news article in 32-64 words and maximum 2 sentences. The length of the summary should be appropriate for the length and complexity of the original text, providing a clear and accurate overview without omitting any important information."
#     }
#   ],
#   temperature=1,
#   max_tokens=1024,
#   top_p=1,
#   frequency_penalty=0,
#   presence_penalty=0
# )

# print(response)