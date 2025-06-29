begin sierra IsothermalCompression15pcf
  
  title Compression under isothermal conditions
  
  begin feti equation solver feti
  end feti equation solver feti
  
  ######################################################################
  ###                                                                ###
  ### Input Deck Description                                         ###
  ###                                                                ###
  ######################################################################
  ### UNITS #####
  #
  ###############
  ## Force       = N
  ## Stress      = Pa
  ## Mass        = Kg
  ## Length      = m
  ## Temperature = K
  ## Time        = s
  ## Density     = kg / m^3
  ###############
  
  ######################################################################
  ###                                                                ###
  ### APREPRO  Inputs  and function definitions                      ###
  ###                                                                ###
  ######################################################################
  #{KELVIN = 273.15}
  #NSTEPS_PER_CYCLE = {NSTEPS_PER_CYCLE = 1000}
  #dt0 = {dt0 = 1.0e-2} # s
  
  
  ### --------------------------------------------------
  ### Aprepro constants specified through the parametric studies
  ### --------------------------------------------------
  #nominal_strain_rate = {nominal_strain_rate = 10.0}
  #max_nominal_strain = {max_nominal_strain = 0.1}
  
  ### --------------------------------------------------
  ### Aprepro constants not specified through the parametric studies
  ### --------------------------------------------------
  
  ## These are the simulation dimensions (its actually the dimensions of the 1/8th part modeled)
  #initial_height = {initial_height = 1} # m
  #initial_area = {initial_area = 1} # m^2
  
  ## (Initial) Time Steps
  
  ### --------------------------------------------------
  ### Derived Aprepro constants
  ### --------------------------------------------------
  
  #displacement_rate = {displacement_rate = nominal_strain_rate * initial_height}
  #max_displacement = {max_displacement = max_nominal_strain * initial_height}
  #cycle_time = {cycle_time = 2 * abs(max_nominal_strain / nominal_strain_rate)}
  #load_time = {load_time = abs(max_nominal_strain / nominal_strain_rate)}
  #end_time = {end_time = 1e3}
  ### --------------------------------------------------
  ### Displacement Rate History
  ### --------------------------------------------------
  begin definition for function applied_disp_fun
    type is piecewise linear
    begin values
      0.0                          0.0
      {load_time}                  {max_displacement}
      {end_time + load_time}       {max_displacement}
    end values
  end definition for function applied_disp_fun
  
  ######################################################################
  ###                                                                ###
  ### Geometry, Section, and Mesh Definitions                           ###
  ###                                                                ###
  ######################################################################
  
  ### --------------------------------------------------
  ### Directions
  ### --------------------------------------------------
  
  define direction x with vector 1.0 0.0 0.0
  define direction y with vector 0.0 1.0 0.0
  define direction z with vector 0.0 0.0 1.0
  
  ### --------------------------------------------------
  ### Material Model Definitions
  ### --------------------------------------------------
  
  begin property specification for material foam
    density = 240.0
    
    begin parameters for model neo_hookean
      bulk modulus = 10.0
      shear modulus = 0.855
    end

    begin parameters for model modularhyperelasticdamage
      #
      # choose a hyperelastic model
      #
      strain energy density = neohookean
      #hyper viscoelastic formulation = ISV
      #neq model = quadraticmandel
      damage model = nodamage
      #
      # hyperelastic properties
      #
      bulk modulus  = 10.0
      shear modulus = 0.855
      #
      # shift factor model
      #
      #shift factor model = none
      #WLF C1 = 17.44
      #WLF C2 = 51.6
      #reference temperature = 60.0
      # 
      # common viscoelastic properties
      #
      #bulk rubbery 0  = 10.0
      #shear rubbery 0 = 0.855
      ##bulk glassy 0   = 10.0
      #shear glassy 0  = 0.855
      # 
      # Prony series
      #
      #num bulk prony terms  = 0
      #num shear prony terms = 1
      #
      #relax time 1 = 10.0
      #
      #f2 1 = 1.0
      #
    end
    
    
  end property specification for material foam
  
  ### --------------------------------------------------
  ### Element Block Definitions
  ### --------------------------------------------------
  
  begin solid section solid_1
    #formulation = fully_integrated
  end solid section solid_1
  
  
  begin finite element model mesh1
    Database Name = mesh/mesh_hex8.g
    Database Type = exodusII
    
    begin parameters for block block_1
      material foam
      model = neo_hookean
      #model = modularhyperelasticdamage
      section = solid_1
      effective moduli model = elastic
    end parameters for block block_1
    
  end finite element model mesh1
  
  ######################################################################
  ###                                                                ###
  ### ADAGIO Procedure                                               ###
  ###                                                                ###
  ######################################################################
  
  begin adagio procedure adagio_procedure
    
    ### --------------------------------------------------
    ### Time Step Definition
    ### --------------------------------------------------
    
    begin time control
      
      begin time stepping block pload
        start time =  0.0
        begin parameters for adagio region adagio_region
          time increment = {dt0}
          #Number of time steps =  {NSTEPS_PER_CYCLE}
        end parameters for adagio region adagio_region
      end time stepping block pload
      
      #termination time = {cycle_time}
      termination time = {end_time}
      #termination time = {end_time + 2.0 * load_time}
    end time control
    
    ######################################################################
    ###                                                                ###
    ### ADAGIO Region                                                  ###
    ###                                                                ###
    ######################################################################
    
    begin adagio region adagio_region
      use finite element model mesh1
      logfile precision = 7
      
      ### --------------------------------------------------
      ### Output Commands
      ### --------------------------------------------------
      
      ######################################################################
      ###                                                                ###
      ### Output Commands                                          ###
      ###                                                                ###
      ######################################################################
      
      
      #User Output
      # The reaction force will be postive in compression for nset_y0
      #begin user output
      #  node set = nset_3
      #  compute global axial_force as sum of nodal reaction(y)
      #  compute at every step
      #end user output
      
      #begin user output
      #  compute global nominal_stress_yy from expression "-axial_force / {initial_area}"
      #  compute at every step
      #end user output
      
      #begin user output
      #  node set = nset_111 # Single node
      ##  compute global delta_x as average of nodal displacement(x)
      #  compute global delta_y as average of nodal displacement(y)
      #  compute global delta_z as average of nodal displacement(z)
      #  compute at every step
      #end user output
      
      #begin user output
      #  compute global nominal_strain_xx from expression "delta_x / {initial_height}"
      #  compute global nominal_strain_yy from expression "delta_y / {initial_height}"
      #  compute global nominal_strain_zz from expression "delta_z / {initial_height}"
      # compute at every step
      #end user output
      
      # Field Output
      begin Results Output output_adagio_region
        Database Name = %B.e
        Database Type = exodusII
        at step 0, increment = 1
        nodal Variables = displacement as displ
        nodal Variables = reaction as reaction_forces
        #element Variables = stress as stress
        #element Variables = unrotated_log_strain as logU
        #element variables = a as shift_factor
        #element variables = fv as Fv
        #element variables = temperature as temperature
        #global variables = nominal_strain_xx as nominal_strain_xx
        #global variables = nominal_strain_yy as nominal_strain_yy
        #global variables = nominal_strain_zz as nominal_strain_zz
        #global variables = nominal_stress_yy as nominal_stress_yy
      end results output output_adagio_region
      
      
      # Heartbeat Output
      begin heartbeat ouput DATAHB
        stream name = %B.hb
        append = off
        precision = 10
        legend = on
        format=CSV #default
        labels=off
        timestamp format=""
        at time 0, increment = 1.0E-6
        global time as time
        global timestep as timestep
        global nominal_strain_xx
        global nominal_strain_yy
        global nominal_strain_zz
        global nominal_stress_yy
      end heartbeat ouput DATAHB
      
      ### --------------------------------------------------
      ### Boundary Conditions
      ### --------------------------------------------------
      
      begin initial temperature
        block = block_1
        magnitude = 75.0
      end

      # Platen Condition Only During Loading
      begin prescribed displacement
        node set = nset_5
        direction = y
        function = applied_disp_fun
        #scale factor = -1.0
      end prescribed displacement
      
      # 3 Symmetry Conditions
      begin fixed displacement
        node set = nset_3
        components = x y z
      end fixed displacement
      
      begin fixed displacement
        node set = nset_5
        components = x z
      end fixed displacement
      
      
      ### --------------------------------------------------
      ### Solver definition
      ### --------------------------------------------------
      
      begin adaptive time stepping
          cutback factor = 0.80
          growth factor = 1.05
          maximum failure cutbacks = 30
          target iterations = 100
          iteration window  = 50
          minimum multiplier = 1.0e-3
          maximum multiplier = {1*ceil(1./dt0)} during pload
          #maximum multiplier = {0.1*ceil(1./dt0)} during ploadcy2
      end adaptive time stepping
      
      begin solver
        
        begin loadstep predictor
          type = scale_factor
          scale factor = 0.0, 1.0
        end loadstep predictor
        
        begin cg
          target relative residual = 1.0e-10
          target residual = 1.0e-6
          acceptable relative residual = 1.0E-7
          acceptable residual = 1.0E-5
          Maximum Iterations = 200
          Line Search secant
          iteration print  = 10
          
          begin full tangent preconditioner
            linear solver = feti
            #tangent diagonal scale = 1e-6
            #iteration update = 50
            maximum updates for modelproblem= 1
            small number of iterations = 50
          end full tangent preconditioner
        end cg
        
      end solver
      
    end adagio region adagio_region
  end adagio procedure adagio_procedure
end sierra IsothermalCompression15pcf
