# /*
# * Title: server for hosting a computer vision model
# * Description: this is a flask server for hosting computer vision model which can identify diff type of crop desies ()
# * Author: Sayed 
# * Date: 21/09/2023
# *
# */

# importing necessary module
import os
from flask import Flask, request, render_template, redirect, url_for, make_response
import base64
# PILfor Image
from PIL import Image
from io import BytesIO
import json
# loiading the learner
from fastcore.all import *
from fastai.learner import load_learner
from flask_cors import CORS
app = Flask(__name__)


# setting up the cors for ignorring the cors policy
CORS(app)


@app.get('/')
def home():
    return {
        "wlc": "something is gonna update soon"
    }

# possible_predictions
# possible_preds = ['Paddy_Blast_(Pyricularia_oryzae)',
#  'Paddy_Sheath_Blight_(Rhizoctonia_solani)',
#  'Paddy_Brown_Spot_(Bipolaris_oryzae)',
#  'Paddy_Rice_Tungro_Virus_(RTV)',
#  'Paddy_Rice_Leaf_Folder_(Cnaphalocrocis_medinalis)',
#  'Paddy_Rice_Hispa_(Dicladispa_armigera)',
#  'Paddy_Bacterial_Leaf_Blight_(Xanthomonas_oryzae_pv._oryzae)',
#  'Paddy_Stem_Borer_(Scirpophaga_spp.)',
#  'Paddy_Rice_Hopper_(Nephotettix_spp.)',
#  'Paddy_Gall_Midge_(Orseolia_oryzae)',
#  'Wheat_Rust_(Puccinia_spp.)',
#  'Wheat_Powdery_Mildew_(Blumeria_graminis)',
#  'Wheat_Fusarium_Head_Blight_(Fusarium_spp.)',
#  'Wheat_Wheat_Streak_Mosaic_Virus_(WSMV)',
#  'Wheat_Yellow_Rust_(Puccinia_striiformis)',
#  'Wheat_Septoria_Leaf_Blotch_(Septoria_tritici)',
#  'Wheat_Wheat_Aphid_(Sitobion_avenae)',
#  'Wheat_Hessian_Fly_(Mayetiola_destructor)',
#  'Wheat_Karnal_Bunt_(Tilletia_indica)',
#  'Wheat_Wheat_Blast_(Magnaporthe_oryzae)',
#  'Potato_Late_Blight_(Phytophthora_infestans)',
#  'Potato_Early_Blight_(Alternaria_solani)',
#  'Potato_Potato_Cyst_Nematode_(Globodera_rostochiensis)',
#  'Potato_Potato_Virus_Y_(PVY)',
#  'Potato_Blackleg_(Pectobacterium_atrosepticum)',
#  'Potato_Potato_Tuber_Moth_(Phthorimaea_operculella)',
#  'Potato_Leaf_Roll_Virus_(PLRV)',
#  'Potato_Common_Scab_(Streptomyces_spp.)',
#  'Potato_Pink_Rot_(Phytophthora_erythroseptica)',
#  'Potato_Bacterial_Wilt_(Ralstonia_solanacearum)',
#  'Jute_Jute_Anthracnose_(Colletotrichum_gloeosporioides)',
#  'Jute_Stem_Rot_(Macrophomina_phaseolina)',
#  'Jute_Leaf_Blight_(Alternaria_spp.)',
#  'Jute_Jute_Yellow_Mosaic_Virus_(JYMV)',
#  'Jute_Damping_Off_(Rhizoctonia_solani)',
#  'Jute_Bacterial_Soft_Rot_(Erwinia_spp.)',
#  'Jute_Jute_Leaf_Miner_(Spodoptera_litura)',
#  'Jute_Aphid_(Aphis_gossypii)',
#  'Jute_Whitefly_(Bemisia_tabaci)',
#  'Jute_Jute_Root_Knot_Nematode_(Meloidogyne_spp.)',
#  'Garlic_White_Rot_(Sclerotium_cepivorum)',
#  'Garlic_Purple_Blotch_(Alternaria_porri)',
#  'Garlic_Downy_Mildew_(Peronospora_destructor)',
#  'Garlic_Rust_(Puccinia_allii)',
#  'Garlic_Neck_Rot_(Botrytis_allii)',
#  'Garlic_Onion_Thrips_(Thrips_tabaci)',
#  'Garlic_Stem_and_Bulb_Nematode_(Ditylenchus_dipsaci)',
#  'Garlic_Garlic_Bloat_Nematode_(Ditylenchus_destructor)',
#  'Garlic_Fusarium_Basal_Rot_(Fusarium_oxysporum)',
#  'Garlic_Garlic_Bulb_Mite_(Rhizoglyphus_spp.)',
#  'Onion_Downy_Mildew_(Peronospora_destructor)',
#  'Onion_Purple_Blotch_(Alternaria_porri)',
#  'Onion_Onion_White_Rot_(Sclerotium_cepivorum)',
#  'Onion_Stem_and_Bulb_Nematode_(Ditylenchus_dipsaci)',
#  'Onion_Onion_Thrips_(Thrips_tabaci)',
#  'Onion_Botrytis_Neck_Rot_(Botrytis_allii)',
#  'Onion_Smut_(Urocystis_cepulae)',
#  'Onion_Fusarium_Basal_Rot_(Fusarium_oxysporum)',
#  'Onion_Iris_Yellow_Spot_Virus_(IYSV)',
#  'Onion_Onion_Bulb_Mite_(Aculops_fuchsiae)',
#  'Corn_(Maize)_Maize_Rust_(Puccinia_spp.)',
#  'Corn_(Maize)_Maize_Streak_Virus_(MSV)',
#  'Corn_(Maize)_Maize_Leaf_Blight_(Cercospora_spp.)',
#  'Corn_(Maize)_Maize_Dwarf_Mosaic_Virus_(MDMV)',
#  'Corn_(Maize)_Northern_Corn_Leaf_Blight_(Exserohilum_turcicum)',
#  'Corn_(Maize)_Corn_Earworm_(Helicoverpa_zea)',
#  'Corn_(Maize)_Fall_Armyworm_(Spodoptera_frugiperda)',
#  'Corn_(Maize)_Corn_Smut_(Ustilago_maydis)',
#  'Corn_(Maize)_Corn_Borer_(Ostrinia_nubilalis)',
#  'Corn_(Maize)_Maize_Weevil_(Sitophilus_zeamais)',
#  'Sugarcane_Red_Rot_(Colletotrichum_falcatum)',
#  'Sugarcane_Smut_(Sporisorium_scitamineum)',
#  'Sugarcane_Sugarcane_Mosaic_Virus_(SCMV)',
#  'Sugarcane_Sugarcane_Yellow_Leaf_Virus_(SCYLV)',
#  'Sugarcane_Bacterial_Wilt_(Erwinia_spp.)',
#  'Sugarcane_White_Grubs_(Holotrichia_spp.)',
#  'Sugarcane_Sugarcane_Woolly_Aphid_(Ceratovacuna_lanigera)',
#  'Sugarcane_Root-Knot_Nematode_(Meloidogyne_spp.)',
#  'Sugarcane_Sugarcane_Scale_(Meliococcus_spp.)',
#  'Sugarcane_Sugarcane_Rust_(Puccinia_kuehnii)',
#  'Tea_Tea_Red_Spider_Mite_(Oligonychus_coffeae)',
#  'Tea_Tea_Mosquito_Bug_(Helopeltis_spp.)',
#  'Tea_Tea_Thrips_(Scirtothrips_dorsalis)',
#  'Tea_Tea_Green_Leafhopper_(Empoasca_spp.)',
#  'Tea_Tea_Pink_Disease_(Corticium_salmonicolor)',
#  'Tea_Tea_Anthracnose_(Colletotrichum_spp.)',
#  'Tea_Tea_Root_Rot_(Phytophthora_spp.)',
#  'Tea_Tea_Grey_Blight_(Pestalotiopsis_spp.)',
#  'Tea_Tea_Red_Leaf_Blister_(Exobasidium_vexans)',
#  'Tea_Tea_Yellow_Mosaic_Virus_(TYMV)']


# defining the predictroue
@app.post('/predict')
def predict():
    """
    things that we should recive
    {
        "image":"base64code",
        "apikey":"NOthing"
    }

    """
    try:
        # loading the data from the user and get image base64 string
        data = json.loads(request.data)
        base64_image_data = data["image"]
        apikey = data["apikey"]

        # checking the api key so that false request cannot be send
        if apikey!="sayedSKC@386":
            return make_response({"error": "401 ", "message": "Unauthorized "}, 401)





        # at first convert the base64 image into image
        try:
            binary_data = base64.b64decode(base64_image_data)
            # image = Image.open(BytesIO(binary_data))
            with open("input_img.jpg", 'wb') as img:
                img.write(binary_data)
            # image.show()
        except (base64.binascii.Error, OSError, Exception) as e:

            return make_response({"error": "500", "message": "Conversion is not poossible"}, 500)

        # now fit to the model and return the result to the user
        learner = load_learner("./skc_enhanced_v2.pkl")
        # result =learner.predict(image)
        pred_class, pred_idx, proabs = learner.predict('input_img.jpg')
        # deleting the image file for cleaning the gubage
        os.unlink('./input_img.jpg')
        pred_labels = searches = ['Paddy_Blast_(Pyricularia_oryzae)',
 'Paddy_Sheath_Blight_(Rhizoctonia_solani)',
 'Paddy_Brown_Spot_(Bipolaris_oryzae)',
 'Paddy_Rice_Tungro_Virus_(RTV)',
 'Paddy_Rice_Leaf_Folder_(Cnaphalocrocis_medinalis)',
 'Paddy_Rice_Hispa_(Dicladispa_armigera)',
 'Paddy_Bacterial_Leaf_Blight_(Xanthomonas_oryzae_pv._oryzae)',
 'Paddy_Stem_Borer_(Scirpophaga_spp.)',
 'Paddy_Rice_Hopper_(Nephotettix_spp.)',
 'Paddy_Gall_Midge_(Orseolia_oryzae)',
 'Wheat_Rust_(Puccinia_spp.)',
 'Wheat_Powdery_Mildew_(Blumeria_graminis)',
 'Wheat_Fusarium_Head_Blight_(Fusarium_spp.)',
 'Wheat_Wheat_Streak_Mosaic_Virus_(WSMV)',
 'Wheat_Yellow_Rust_(Puccinia_striiformis)',
 'Wheat_Septoria_Leaf_Blotch_(Septoria_tritici)',
 'Wheat_Wheat_Aphid_(Sitobion_avenae)',
 'Wheat_Hessian_Fly_(Mayetiola_destructor)',
 'Wheat_Karnal_Bunt_(Tilletia_indica)',
 'Wheat_Wheat_Blast_(Magnaporthe_oryzae)',
 'Potato_Late_Blight_(Phytophthora_infestans)',
 'Potato_Early_Blight_(Alternaria_solani)',
 'Potato_Potato_Cyst_Nematode_(Globodera_rostochiensis)',
 'Potato_Potato_Virus_Y_(PVY)',
 'Potato_Blackleg_(Pectobacterium_atrosepticum)',
 'Potato_Potato_Tuber_Moth_(Phthorimaea_operculella)',
 'Potato_Leaf_Roll_Virus_(PLRV)',
 'Potato_Common_Scab_(Streptomyces_spp.)',
 'Potato_Pink_Rot_(Phytophthora_erythroseptica)',
 'Potato_Bacterial_Wilt_(Ralstonia_solanacearum)',
 'Jute_Jute_Anthracnose_(Colletotrichum_gloeosporioides)',
 'Jute_Stem_Rot_(Macrophomina_phaseolina)',
 'Jute_Leaf_Blight_(Alternaria_spp.)',
 'Jute_Jute_Yellow_Mosaic_Virus_(JYMV)',
 'Jute_Damping_Off_(Rhizoctonia_solani)',
 'Jute_Bacterial_Soft_Rot_(Erwinia_spp.)',
 'Jute_Jute_Leaf_Miner_(Spodoptera_litura)',
 'Jute_Aphid_(Aphis_gossypii)',
 'Jute_Whitefly_(Bemisia_tabaci)',
 'Jute_Jute_Root_Knot_Nematode_(Meloidogyne_spp.)',
 'Garlic_White_Rot_(Sclerotium_cepivorum)',
 'Garlic_Purple_Blotch_(Alternaria_porri)',
 'Garlic_Downy_Mildew_(Peronospora_destructor)',
 'Garlic_Rust_(Puccinia_allii)',
 'Garlic_Neck_Rot_(Botrytis_allii)',
 'Garlic_Onion_Thrips_(Thrips_tabaci)',
 'Garlic_Stem_and_Bulb_Nematode_(Ditylenchus_dipsaci)',
 'Garlic_Garlic_Bloat_Nematode_(Ditylenchus_destructor)',
 'Garlic_Fusarium_Basal_Rot_(Fusarium_oxysporum)',
 'Garlic_Garlic_Bulb_Mite_(Rhizoglyphus_spp.)',
 'Onion_Downy_Mildew_(Peronospora_destructor)',
 'Onion_Purple_Blotch_(Alternaria_porri)',
 'Onion_Onion_White_Rot_(Sclerotium_cepivorum)',
 'Onion_Stem_and_Bulb_Nematode_(Ditylenchus_dipsaci)',
 'Onion_Onion_Thrips_(Thrips_tabaci)',
 'Onion_Botrytis_Neck_Rot_(Botrytis_allii)',
 'Onion_Smut_(Urocystis_cepulae)',
 'Onion_Fusarium_Basal_Rot_(Fusarium_oxysporum)',
 'Onion_Iris_Yellow_Spot_Virus_(IYSV)',
 'Onion_Onion_Bulb_Mite_(Aculops_fuchsiae)',
 'Corn_(Maize)_Maize_Rust_(Puccinia_spp.)',
 'Corn_(Maize)_Maize_Streak_Virus_(MSV)',
 'Corn_(Maize)_Maize_Leaf_Blight_(Cercospora_spp.)',
 'Corn_(Maize)_Maize_Dwarf_Mosaic_Virus_(MDMV)',
 'Corn_(Maize)_Northern_Corn_Leaf_Blight_(Exserohilum_turcicum)',
 'Corn_(Maize)_Corn_Earworm_(Helicoverpa_zea)',
 'Corn_(Maize)_Fall_Armyworm_(Spodoptera_frugiperda)',
 'Corn_(Maize)_Corn_Smut_(Ustilago_maydis)',
 'Corn_(Maize)_Corn_Borer_(Ostrinia_nubilalis)',
 'Corn_(Maize)_Maize_Weevil_(Sitophilus_zeamais)',
 'Sugarcane_Red_Rot_(Colletotrichum_falcatum)',
 'Sugarcane_Smut_(Sporisorium_scitamineum)',
 'Sugarcane_Sugarcane_Mosaic_Virus_(SCMV)',
 'Sugarcane_Sugarcane_Yellow_Leaf_Virus_(SCYLV)',
 'Sugarcane_Bacterial_Wilt_(Erwinia_spp.)',
 'Sugarcane_White_Grubs_(Holotrichia_spp.)',
 'Sugarcane_Sugarcane_Woolly_Aphid_(Ceratovacuna_lanigera)',
 'Sugarcane_Root-Knot_Nematode_(Meloidogyne_spp.)',
 'Sugarcane_Sugarcane_Scale_(Meliococcus_spp.)',
 'Sugarcane_Sugarcane_Rust_(Puccinia_kuehnii)',
 'Tea_Tea_Red_Spider_Mite_(Oligonychus_coffeae)',
 'Tea_Tea_Mosquito_Bug_(Helopeltis_spp.)',
 'Tea_Tea_Thrips_(Scirtothrips_dorsalis)',
 'Tea_Tea_Green_Leafhopper_(Empoasca_spp.)',
 'Tea_Tea_Pink_Disease_(Corticium_salmonicolor)',
 'Tea_Tea_Anthracnose_(Colletotrichum_spp.)',
 'Tea_Tea_Root_Rot_(Phytophthora_spp.)',
 'Tea_Tea_Grey_Blight_(Pestalotiopsis_spp.)',
 'Tea_Tea_Red_Leaf_Blister_(Exobasidium_vexans)',
 'Tea_Tea_Yellow_Mosaic_Virus_(TYMV)']
        proabs = proabs.numpy().tolist()
        
        return {
            "output": {
                "pred_class": pred_class,
                "proab_labels": pred_labels,
                "proabs": proabs

            }
        }
    except:
        # handeling any else case
        return  make_response({"status": 500, "message": "internal server error"}, 500)




# Setting up the default 404 error handler
@app.errorhandler(404)
def page_not_found(error):
    return make_response({"status": 404, "message": "page not found"}, 404)


# finally run the app
if __name__ == "__main__":
    app.run(debug=False)
