pipeline {
    agent any

    stages {
        stage('Testing') {
            steps {
                sh ./scripts/test_daint.sh
            }
        }
    }
}
