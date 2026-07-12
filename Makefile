# Управление проектом. Требует GNU make.
# Если make не установлен (обычная ситуация на Windows) — используйте
# те же docker compose команды напрямую, см. раздел «Запуск» в README.md.
COMPOSE = docker compose -f docker/docker-compose.yml --profile ml --profile airflow

.PHONY: build up down restart logs ps run status clean

build:            ## собрать образы (первый раз тянет базовые, ~несколько ГБ)
	$(COMPOSE) build

up:               ## поднять весь стек: HDFS + Spark + trainer + Airflow
	$(COMPOSE) up -d

down:             ## остановить всё (данные в HDFS сохранятся)
	$(COMPOSE) down

restart:          ## перезапустить стек
	$(COMPOSE) restart

ps:               ## статус контейнеров
	$(COMPOSE) ps

logs:             ## логи всех сервисов
	$(COMPOSE) logs -f

run:              ## запустить пайплайн (Airflow DAG)
	docker exec airflow-scheduler airflow dags unpause credit_scoring
	docker exec airflow-scheduler airflow dags trigger credit_scoring

status:           ## статус последнего прогона пайплайна
	docker exec airflow-scheduler airflow dags list-runs -d credit_scoring -o plain | head -3

clean:            ## down + УДАЛИТЬ тома (данные HDFS сотрутся!)
	$(COMPOSE) down -v
