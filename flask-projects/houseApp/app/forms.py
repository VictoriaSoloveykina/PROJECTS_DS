from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField,HiddenField
from wtforms.validators import DataRequired

class EditForm(FlaskForm):
    id = HiddenField("flatId")
    title = StringField('Заголовок объявления')
    city = StringField('Город')
    rooms = IntegerField('Количество комнат')
    area = StringField('Площадь')
    floor = IntegerField('Этаж')
    floors_total = IntegerField('Количество этажей в доме')
    cost = IntegerField('Стоимость')
    submit = SubmitField('Сохранить')
