from django.db import models

def get_up_to(instance, filename):
    return f'imgs/{instance.formation.nom}_{instance.formation.specialite}/{instance.nom}_{instance.prenom}/{filename}'


class Formation(models.Model):
    nom = models.CharField(max_length=200, default='')
    specialite = models.CharField(max_length=200, default='')
    nom_chef_formation = models.CharField(max_length=200, default='hamza eljokh')
    mail_chef_formation = models.EmailField(max_length=200, default='hamza.eljokh@gmail.com')

    def __str__(self):
        return self.nom + ' ' + self.specialite


class Etudiant(models.Model):
    nom = models.CharField(max_length=20)
    prenom = models.CharField(max_length=20)
    date_naissance = models.DateTimeField('date de naissance')
    formation = models.ForeignKey(Formation, on_delete=models.CASCADE)
    photo1 = models.ImageField(upload_to=get_up_to, blank=True)
    photo2 = models.ImageField(upload_to=get_up_to, blank=True)
    photo3 = models.ImageField(upload_to=get_up_to, blank=True)
    photo4 = models.ImageField(upload_to=get_up_to, blank=True)
    photo5 = models.ImageField(upload_to=get_up_to, blank=True)
    photo6 = models.ImageField(upload_to=get_up_to, blank=True)
    photo7 = models.ImageField(upload_to=get_up_to, blank=True)
    photo8 = models.ImageField(upload_to=get_up_to, blank=True)
    photo9 = models.ImageField(upload_to=get_up_to, blank=True)
    photo10 = models.ImageField(upload_to=get_up_to, blank=True)

    def __str__(self):
        return 'Etudiant : '+ self.nom + ' ' + self.prenom + '-' + self.formation.nom

class Seance(models.Model):
    date = models.DateField('date du Seance', null=True)
    time_debut = models.TimeField('temps de debut', null=True)
    time_fin = models.TimeField('temps de fin', null=True)
    numero_classe = models.CharField(max_length=20)
    formation = models.ForeignKey(Formation, on_delete=models.CASCADE)

    def __str__(self):
        return 'class : ' + self.numero_classe + ' commence le : ' + str(self.date) + ' à ' + str(self.time_debut)


class Absance(models.Model):
    seance = models.ForeignKey(Seance, on_delete=models.CASCADE)
    etudiant = models.ForeignKey(Etudiant, on_delete=models.CASCADE)

    def __str__(self):
        return self.etudiant.nom + ' ' + self.etudiant.prenom + 'absent le' + str(self.seance.date) + str(self.seance.time_debut)



