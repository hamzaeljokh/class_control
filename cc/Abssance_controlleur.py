from threading import Thread
from cc.dnn import dnn
from .models import Etudiant, Seance, Absance, Formation
from datetime import timedelta, datetime, date
from django.shortcuts import get_object_or_404
from pytz import timezone as tz2
import time

#from django.db.models import Q

class AbssanceControlleur(Thread):
    """ Classe controlleur das abssances"""

    def __init__(self, formation, seconde_to_slepp):
        self.seconde_to_slepp = seconde_to_slepp
        formation_to_control = formation.nom + '_' + formation.specialite
        self.formation = formation
        self.d = dnn(formation_to_control)
        Thread.__init__(self)

    def run(self):
        """Code à exécuter pendant l'exécution du thread."""

        Etudiant_Q_set = Etudiant.objects.filter(formation__id=self.formation.id)
        etudiants = set(e.nom+'_'+e.prenom for e in Etudiant_Q_set)
        print('tous les etudiant de la formation',etudiants)

        while True:#tant que il ya encore des seance à venir = more_seance.count() >= 1
            Seances_couran = Seance.objects.filter(date=date.today(),
                                                   time_debut__lte=datetime.now(tz2('CET')),
                                                   time_fin__gte=datetime.now(tz2('CET')),
                                                   formation__id=self.formation.id, )
            last_Seance_couran_id = 1
            next_Seance_couran_id = 1
            Test_Fait = False
            etudiant_présent = set()
            if Seances_couran.count() >= 1:
                temp_s= Seances_couran[0]
                print("CC agent start , seance : de ", str(temp_s.time_debut), "à", str(temp_s.time_fin),
                      "le", str(temp_s.date))

            # tant que on est dans une séance courant, et pour la même séance
            while Seances_couran.count() >= 1 and last_Seance_couran_id==next_Seance_couran_id:
                Test_Fait = True
                Seance_couran = Seances_couran[0]
                last_Seance_couran_id = Seance_couran.id

                try:
                    etudiant_deteter = self.d.start_controle()
                    print("etudiant detecter dans ce sub-control:", etudiant_deteter)
                    for e in etudiant_deteter:
                        etudiant_présent.add(e)
                    time.sleep(self.seconde_to_slepp)
                    Seances_couran = Seance.objects.filter(date=date.today(),
                                                            time_debut__lte=datetime.now(tz2('CET')),
                                                            time_fin__gte=datetime.now(tz2('CET')),
                                                            formation__id=self.formation.id, )
                    if Seances_couran.count() >= 1:
                        next_Seance_couran_id = Seances_couran[0].id
                    else:
                        next_Seance_couran_id = 0#just n'import quoi
                except Exception as e:
                    print("######## erreur loop thread")
                    #print(e.args)
                    time.sleep(int(self.seconde_to_slepp / 5))
                    pass

            if Test_Fait:
                print("total des etudiants detecter présent dans la seance :","CC agent start , seance : de ", str(Seance_couran.time_debut), "à", str(Seance_couran.time_fin),
                          "le", str(Seance_couran.date), "::", etudiant_présent)
                print("Abssances:",etudiants-etudiant_présent)
                etudiants_absant = etudiants-etudiant_présent
                for etudiant_absant in etudiants_absant:
                    n = etudiant_absant.split('_')[0]
                    p = etudiant_absant.split('_')[1]
                    etu = get_object_or_404(Etudiant, nom=n, prenom=p)
                    A=Absance(seance=Seance_couran, etudiant=etu)
                    A.save()
            else:
                print("aucune séance en cours")
                time.sleep(self.seconde_to_slepp*2)

            # more_seance_q = Q(date=date.today()) & Q(formation__id=self.formation.id) & \
            #                 Q(Q(time_debut__gte=datetime.now(tz2('CET'))) | Q(
            #                     Q(time_fin__gte=datetime.now(tz2('CET'))) & Q(
            #                         time_debut__lte=datetime.now(tz2('CET')))))
            # # query or Q
            # more_seance = Seance.objects.filter(more_seance_q)
