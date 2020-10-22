from .models import Etudiant, Formation, Absance
from django.shortcuts import get_object_or_404, render
from cc.Abssance_controlleur import AbssanceControlleur
from cc.dnn import dnn

started = False


def index(request):
    Etudiant_list = Etudiant.objects.all()
    formation = get_object_or_404(Formation, id=1)
    context = {'Etudiant_list': Etudiant_list, 'formation': formation}
    return render(request, 'index.html', context)


def formations_list(request):
    formation_list = Formation.objects.all()
    context = {'formation_list': formation_list}
    return render(request, 'formation.html', context)


def get_etudiants_list(request, id_formation):
    print(id_formation)
    print(type(id_formation))
    Etudiant_list = Etudiant.objects.filter(formation__id=id_formation)
    #formation = Formation.objects.get(id=id_formation)
    formation = get_object_or_404(Formation, id=id_formation)
    print(formation.nom)
    context = {'Etudiant_list': Etudiant_list, 'formation': formation}
    return render(request, 'index.html', context)


#liste des abssance pour une durée et une formation
def absances_list(request):#, date1, date2
    formation = None
    absances = None
    if request.POST.get("dated", "") and request.POST.get("datef", "") and request.POST.get("formation_id", ""):
        try:
            print(request.POST['dated'])
            print(request.POST['datef'])
            print(request.POST['formation_id'])
            formation  = get_object_or_404(Formation, id=request.POST['formation_id'])
            absances = Absance.objects.filter(seance__formation__id=request.POST['formation_id'],
                                              seance__date__gte=request.POST['dated'],
                                              seance__date__lte=request.POST['datef'])
        except Exception as e:
            print(e.args)
    elif request.POST.get("formation_id", ""):
        try:
            print(request.POST['formation_id'])
            formation  = get_object_or_404(Formation, id=request.POST['formation_id'])
            absances = Absance.objects.filter(seance__formation__id=request.POST['formation_id'])
        except Exception as e:
            print(e.args)

    formations = Formation.objects.all()
    context = {'formations': formations,
               "absances": absances,
               'formation': formation}

    return render(request, 'abssances.html', context)


def run_control(request, id_formation):
    global started
    if not started:
        formation = get_object_or_404(Formation, id=id_formation)
        controller = AbssanceControlleur(formation, 20)
        controller.start()
        started = True
    formation_list = Formation.objects.all()
    context = {'formation_list': formation_list}
    return render(request, 'formation.html', context)


def train(request, id_formation):
    formation = get_object_or_404(Formation, id=id_formation)
    dir_formation = formation.nom+'_'+formation.specialite
    d = dnn(dir_formation)
    d.train_classifier(5000)
    del d
    formation_list = Formation.objects.all()
    context = {'formation_list': formation_list}
    return render(request, 'formation.html', context)

