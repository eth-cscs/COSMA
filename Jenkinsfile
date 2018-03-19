pipeline {
    agent daint

    stages {
        stage('Testing') {
            steps {
                sh './scripts/test_daint.sh'
            }
        }
    }
}
